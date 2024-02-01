"""
Class for biomedical entity etl
"""
from typing import Sequence
from prisma.models import BiomedicalEntity
from prisma.enums import BiomedicalEntityType, Source
from prisma.types import (
    BiomedicalEntityCreateInput,
    BiomedicalEntityCreateWithoutRelationsInput,
    BiomedicalEntityUpdateInput,
    BiomedicalEntityWhereUniqueInput,
)
from pydash import compact, flatten, group_by, is_empty, omit
import logging

from clients.low_level.prisma import batch_update, prisma_client
from core.ner.cleaning import CleanFunction
from core.ner.linker.types import CandidateSelectorType
from core.ner.normalizer import TermNormalizer
from core.ner.types import CanonicalEntity
from data.domain.biomedical.umls import tuis_to_entity_type
from data.etl.types import RelationIdFieldMap
from typings.documents.common import ENTITY_MAP_TABLES
from utils.list import merge_nested

from .umls.umls_load import UmlsLoader
from ..base_entity_etl import BaseEntityEtl

DEFAULT_TYPE_FIELD = "default_type"
OVERRIDE_TYPE_FIELD = "type"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BiomedicalEntityEtl(BaseEntityEtl):
    """
    Class for biomedical entity etl

    - canonicalizes/normalizes terms
    - creates records for entities and corresponding relationships (e.g. parents, comprised_of)
    """

    def __init__(
        self,
        candidate_selector: CandidateSelectorType,
        relation_id_field_map: RelationIdFieldMap,
        non_canonical_source: Source,
        additional_cleaners: Sequence[CleanFunction] = [],
    ):
        self.normalizer = TermNormalizer(
            candidate_selector=candidate_selector,
            additional_cleaners=additional_cleaners,
        )
        self.relation_id_field_map = relation_id_field_map
        self.non_canonical_source = non_canonical_source

    def _generate_lookup_map(
        self, terms: Sequence[str], vectors: Sequence[Sequence[float]] | None = None
    ) -> dict[str, CanonicalEntity]:
        """
        Generate canonical map for source terms
        """

        lookup_docs = self.normalizer.normalize_strings(terms, vectors)

        # map for quick lookup of canonical entities
        lookup_map = {
            on: de.canonical_entity
            for on, de in zip(terms, lookup_docs)
            if de.canonical_entity is not None
        }
        return lookup_map

    def _generate_crud(
        self,
        terms_to_insert: Sequence[str],
        source_map: dict[str, dict],
        canonical_map: dict[str, CanonicalEntity],
    ) -> tuple[
        list[BiomedicalEntityCreateWithoutRelationsInput],
        list[BiomedicalEntityUpdateInput],
    ]:
        """
        Create record dicts for entity insert

        Args:
            terms_to_insert (Sequence[str]): terms to insert
            source_map (dict): map of "term" to source record for additional fields, e.g. synonyms, active_ingredients, etc.
            canonical_map (dict): map of "term" to canonical entity
        """

        def create_input(orig_name: str) -> BiomedicalEntityCreateInput | None:
            """
            Form create input for a given term
            """
            source_rec = source_map[orig_name]
            canonical = canonical_map.get(orig_name)

            if canonical is not None:
                core_fields = BiomedicalEntityCreateInput(
                    canonical_id=canonical.id or canonical.name.lower(),
                    entity_type=source_rec.get(OVERRIDE_TYPE_FIELD) or canonical.type,
                    is_priority=source_rec.get("is_priority") or False,
                    name=canonical.name.lower(),
                    sources=[Source.UMLS],
                )
            else:
                core_fields = BiomedicalEntityCreateInput(
                    canonical_id=orig_name,
                    entity_type=(
                        source_rec.get(DEFAULT_TYPE_FIELD)
                        or BiomedicalEntityType.UNKNOWN
                    ),
                    is_priority=source_rec.get("is_priority") or False,
                    name=orig_name,
                    sources=[self.non_canonical_source],
                )

            # if the entity type is unknown, don't create the record
            if core_fields["entity_type"] == BiomedicalEntityType.UNKNOWN:
                logger.info("Skipping unknown entity type: %s", orig_name)
                return None

            # create input for N-to-N relationships (synonyms, comprised_of, parents)
            relation_fields = {
                field: spec.form_prisma_relation(source_rec, canonical_map)
                for field, spec in self.relation_id_field_map.items
            }

            return BiomedicalEntityCreateInput(
                **core_fields,
                # include only non-empty relation fields
                **{f: v for f, v in relation_fields.items() if not is_empty(v)},
            )

        # merge records with same canonical id
        flat_recs = compact([create_input(name) for name in terms_to_insert])
        grouped_recs = group_by(flat_recs, "canonical_id")
        update_data: list[BiomedicalEntityUpdateInput] = flatten(
            [
                [BiomedicalEntityUpdateInput(**merge_nested(*groups))]  # type: ignore
                if canonical_id is None
                else groups
                for canonical_id, groups in grouped_recs.items()
            ]
        )

        # for create, remove the relationships. create must happen before update
        # because update includes linking biomedical entities
        create_data = [
            BiomedicalEntityCreateWithoutRelationsInput(
                **omit(r, *self.relation_id_field_map.fields)  # type: ignore
            )
            for r in update_data
        ]

        return create_data, update_data

    @staticmethod
    def get_update_where(
        rec: BiomedicalEntityUpdateInput | BiomedicalEntityCreateInput,
    ) -> BiomedicalEntityWhereUniqueInput:
        """
        Get where clause for update
        - prefer canonical_id, but use name/type if not available
        """
        if rec.get("canonical_id") is not None:
            return {"canonical_id": rec.get("canonical_id") or ""}
        elif rec.get("name") is not None:
            return {
                "name_entity_type": {
                    "name": rec.get("name") or "",
                    "entity_type": rec.get("entity_type")
                    or BiomedicalEntityType.UNKNOWN,
                }
            }

        raise ValueError("Must have canonical_id or name")

    async def copy_all(
        self,
        terms: Sequence[str],
        terms_to_canonicalize: Sequence[str] | None,
        vectors_to_canonicalize: Sequence[Sequence[float]] | None,
        source_map: dict[str, dict],
    ):
        """
        Create records for entities and relationships

        Args:
            terms (Sequence[str]): terms to insert
            terms_to_canonicalize (Sequence[str]): terms to canonicalize
            vectors_to_canonicalize (Sequence[Sequence[float]]): vectors to canonicalize, if we have them
            source_map (dict): map of "term" to source record for additional fields, e.g. synonyms, "active_ingredients", etc.
            force_update (bool): force update of records (otherwise just create & incidental updates for relationships)
        """
        canonical_map = self._generate_lookup_map(
            terms_to_canonicalize or terms, vectors_to_canonicalize
        )
        create_data, upsert_data = self._generate_crud(terms, source_map, canonical_map)

        client = await prisma_client(600)
        # create first, because updates include linking between biomedical entities
        await BiomedicalEntity.prisma(client).create_many(
            data=create_data,
            skip_duplicates=True,
        )

        # then update records with relationships
        await batch_update(
            upsert_data,
            update_func=lambda r, tx: BiomedicalEntity.prisma(tx).update(
                data=BiomedicalEntityUpdateInput(**r),
                where=self.get_update_where(r),
            ),
            batch_size=1000,
        )

    @staticmethod
    async def _update_search_index():
        """
        update search index
        """
        client = await prisma_client(300)
        await client.execute_raw("DROP INDEX IF EXISTS biomedical_entity_search_idx")
        await client.execute_raw(
            f"""
            WITH synonym as (
                SELECT entity_id, array_agg(term) as terms
                FROM entity_synonym
                GROUP BY entity_id
            )
            UPDATE biomedical_entity SET search = to_tsvector('english', name || ' ' || array_to_string(synonym.terms, ' '))
                FROM synonym WHERE entity_id=biomedical_entity.id;
            """
        )
        await client.execute_raw(
            "CREATE INDEX biomedical_entity_search_idx ON biomedical_entity USING GIN(search)"
        )

    @staticmethod
    async def _map_umls():
        """
        Map UMLS to entities, based on id field
            (which we know to be a pipe-delimited list of UMLS ids (+random strings we ignore))

        TODO: should do the mapping insert inline, since we have the ids
        """
        query = """
            INSERT into _entity_to_umls ("A", "B")
            SELECT s.id, cid FROM (
                SELECT id, unnest(string_to_array(canonical_id, '|')) as cid
                FROM biomedical_entity
                WHERE canonical_id IS NOT null
            ) s
            JOIN umls ON umls.id = s.cid
            ON CONFLICT DO NOTHING;
        """
        client = await prisma_client(600)
        await client.execute_raw(query)

    @staticmethod
    async def _create_biomedical_entity_ancestors(n_depth: int = 3):
        """
        Create ancestor biomedical entities from UMLS

        Args:
            n_depth (int): numer of iterations / recursion depth
                e.g. iteration 1, we get parents. iteration 2, we get parents of parents, etc.

        TODO: turn this into an in-memory function and insert only once
        """

        async def execute(n: int):
            # restrict iteration >=1 to entries with parents
            # (since that is a superset of any possible additions)
            has_children_restriction = (
                """AND umls.id in (select "A" from _entity_to_parent where "A" = umls.id)"""
                if n > 0
                else ""
            )
            query = f"""
                SELECT
                    biomedical_entity.id AS child_id,
                    umls_parent.id AS canonical_id,
                    umls_parent.preferred_name AS name,
                    umls_parent.type_ids AS tuis
                FROM umls, _entity_to_umls etu, umls as umls_parent, biomedical_entity
                WHERE
                    etu."A"=biomedical_entity.id
                    AND umls.id=etu."B"
                    AND umls_parent.id=umls.rollup_id
                    {has_children_restriction}
            """
            client = await prisma_client(300)
            results = await client.query_raw(query)
            records = [
                BiomedicalEntityCreateInput(
                    canonical_id=r["canonical_id"],
                    children={"connect": [{"id": r["child_id"]}]},
                    entity_type=tuis_to_entity_type(r["tuis"]),
                    name=r["name"],
                    sources=[Source.UMLS],
                    umls_entities={"connect": [{"id": r["canonical_id"]}]},
                )
                for r in results
            ]

            await batch_update(
                records,
                update_func=lambda r, tx: BiomedicalEntity.prisma(tx).upsert(
                    data={
                        "create": r,
                        "update": BiomedicalEntityUpdateInput(**r),  # type: ignore
                    },
                    where=BiomedicalEntityEtl.get_update_where(r),
                ),
                batch_size=10000,
            )

        for n in range(n_depth):
            logger.info("Adding UMLS as biomedical entity parents, depth %s", n)
            await execute(n)

    @staticmethod
    async def add_counts():
        """
        add counts to biomedical_entity (used for autocomplete ordering)
        """
        client = await prisma_client(6)
        # add counts to biomedical_entity & owner
        for table in ENTITY_MAP_TABLES:
            await client.execute_raw("CREATE TEMP TABLE temp_count(id int, count int)")
            await client.execute_raw(
                f"INSERT INTO temp_count (id, count) SELECT entity_id as id, count(*) FROM {table} GROUP BY entity_id"
            )
            await client.execute_raw(
                "UPDATE biomedical_entity ct SET count=temp_count.count FROM temp_count WHERE temp_count.id=ct.id"
            )
            await client.execute_raw("DROP TABLE IF EXISTS temp_count;")

    @staticmethod
    async def link_to_documents():
        """
        - Link mapping tables "intervenable" and "indicatable" to canonical entities
        - add instance_rollup and category_rollups
        """

        def get_query(table: str) -> str:
            return f"""
                UPDATE {table}
                SET
                    entity_id=entity_synonym.entity_id,
                    canonical_name=lower(biomedical_entity.name),
                    canonical_type=biomedical_entity.entity_type
                FROM entity_synonym, biomedical_entity
                WHERE {table}.name=entity_synonym.term
                AND entity_synonym.entity_id=biomedical_entity.id
            """

        client = await prisma_client(600)
        for table in ["intervenable", "indicatable"]:
            query = get_query(table)
            await client.execute_raw(query)

    @staticmethod
    async def set_rollups():
        """
        Set instance_rollup and category_rollup for map tables (intervenable, indicatable)
        intentional denormalization for reporting (faster than recursive querying)
        """

        def get_query(table: str) -> str:
            # logic is in how rollup_id is set (umls etl / ancestor_selector)
            return f"""
                UPDATE {table}
                SET
                    instance_rollup=lower(instance_rollup_name),
                    category_rollup=lower(category_rollup_name)
                FROM biomedical_entity as main_entity
                JOIN (
                    SELECT
                        etp."B" as main_entity_id,
                        max(id order by is_priority desc) as id,
                        max(name order by is_priority desc) as instance_rollup_name
                    FROM _entity_to_parent etp
                    JOIN biomedical_entity ON id=etp."A" -- "A" is parent
                    GROUP BY etp."B" -- "B" is child
                ) instance_rollup ON main_entity_id=main_entity.id
                LEFT JOIN (
                    SELECT
                        etp."B" as instance_rollup_id,
                        max(name order by is_priority desc) as category_rollup_name
                    FROM _entity_to_parent etp
                    JOIN biomedical_entity ON id=etp."A" -- "A" is parent
                    GROUP BY etp."B" -- "B" is child
                ) category_rollup ON instance_rollup_id=instance_rollup.id
                WHERE {table}.entity_id=main_entity.id
            """

        client = await prisma_client(600)
        for table in ["intervenable", "indicatable"]:
            query = get_query(table)
            await client.execute_raw(query)

    @staticmethod
    async def pre_doc_finalize():
        """
        Finalize etl
        """
        # map umls to entities
        await BiomedicalEntityEtl._map_umls()

        # populate search index with name & syns
        await BiomedicalEntityEtl._update_search_index()

    @staticmethod
    async def post_doc_finalize():
        """
        Run after:
            1) UMLS is loaded
            2) all biomedical entities are loaded
            3) all documents are loaded with corresponding mapping tables (intervenable, indicatable)
        """
        logger.info("Finalizing biomedical entity etl")
        # await BiomedicalEntityEtl.link_to_documents()
        # await BiomedicalEntityEtl.add_counts()

        # perform final UMLS updates, which depends upon Biomedical Entities being in place.
        # NOTE: this will use a filesystem-cached graph if available, which could be stale.
        # await UmlsLoader.set_ontology_levels()

        # recursively add biomedical entity parents (from UMLS)
        await BiomedicalEntityEtl._create_biomedical_entity_ancestors()

        # set instance & category rollups
        await BiomedicalEntityEtl.set_rollups()
