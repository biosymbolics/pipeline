"""
Class for biomedical entity etl
"""

import asyncio
from typing import Sequence
import uuid
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
from constants.umls import UMLS_COMMON_BASES
from core.ner.cleaning import CleanFunction
from core.ner.linker.types import CandidateSelectorType
from core.ner.normalizer import TermNormalizer
from core.ner.types import CanonicalEntity
from data.domain.biomedical.umls import tuis_to_entity_type
from data.etl.types import RelationIdFieldMap
from typings.documents.common import ENTITY_MAP_TABLES
from utils.file import save_as_pickle
from utils.list import merge_nested

from .umls.load_umls import UmlsLoader
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
            if de.canonical_entity is not None  # shouldn't happen
        }
        save_as_pickle(lookup_map, f"canonical_map-{uuid.uuid4()}.pkl")
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
            canonical = canonical_map.get(orig_name) or CanonicalEntity(
                orig_name.lower(), orig_name.lower()
            )

            entity_type = (
                source_rec.get(OVERRIDE_TYPE_FIELD)
                or (
                    canonical.type
                    if canonical.type != BiomedicalEntityType.UNKNOWN
                    else source_rec.get(DEFAULT_TYPE_FIELD)
                )
                or BiomedicalEntityType.UNKNOWN
            )

            # if it is UMLS and we don't have an entity type, skip it (garbage removal)
            if not canonical.is_fake and entity_type == BiomedicalEntityType.UNKNOWN:
                logger.info("Skipping unknown entity type for %s", orig_name)
                return None

            core_fields = BiomedicalEntityCreateInput(
                canonical_id=canonical.id or canonical.name.lower(),
                entity_type=entity_type,
                is_priority=source_rec.get("is_priority") or False,
                name=canonical.name.lower(),
                sources=[
                    self.non_canonical_source if canonical.is_fake else Source.UMLS
                ],
            )

            # create input for N-to-N relationships (synonyms, comprised_of, parents)
            # TODO: really need connectOrCreate for synonyms!
            # https://github.com/RobertCraigie/prisma-client-py/issues/754
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
                (
                    [BiomedicalEntityUpdateInput(**merge_nested(*groups))]  # type: ignore
                    if canonical_id is None
                    else groups
                )
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
        - prefer name (so we can update canonical_id if better match)
        - otherwise, use canonical_id
        """
        if rec.get("name") is not None:
            return {"name": rec.get("name") or ""}
        if rec.get("canonical_id") is not None:
            return {"canonical_id": rec.get("canonical_id") or ""}

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
        logger.info("Updating biomedical_entity search index")
        client = await prisma_client(600)
        await client.execute_raw("DROP INDEX IF EXISTS biomedical_entity_search_idx;")
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
            "CREATE INDEX biomedical_entity_search_idx ON biomedical_entity USING GIN(search);"
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
        logger.info("Mapping entities to UMLS concepts")
        client = await prisma_client(600)
        await client.execute_raw(query)

    @staticmethod
    async def _create_biomedical_entity_ancestors(n_depth: int = 2):
        """
        Create ancestor biomedical entities from UMLS

        Args:
            n_depth (int): numer of iterations / recursion depth
                e.g. iteration 1, we get parents. iteration 2, we get parents of parents, etc.

        TODO: turn this into an in-memory function and insert only once
        """

        async def execute():
            query = f"""
                SELECT
                    child_entity.id AS child_id,
                    umls_parent.id AS canonical_id,
                    umls_parent.preferred_name AS name,
                    umls_parent.type_ids AS type_ids
                FROM
                    umls,
                    _entity_to_umls etu,
                    umls as umls_parent,
                    biomedical_entity as child_entity
                WHERE
                    etu."A"=child_entity.id
                    AND umls.id=etu."B"
                    AND umls_parent.id=umls.rollup_id
                    -- only get records for which a parent with this child doesn't yet exist
                    AND NOT EXISTS (
                        SELECT 1
                        FROM biomedical_entity possible_parent_entity, _entity_to_parent etp
                        WHERE possible_parent_entity.id=etp."A"
                        AND child_entity.id=etp."B"
                        AND canonical_id=umls_parent.id
                    )
            """
            client = await prisma_client(300)
            results = await client.query_raw(query)
            records = [
                BiomedicalEntityCreateInput(
                    canonical_id=r["canonical_id"],
                    children={"connect": [{"id": r["child_id"]}]},
                    entity_type=tuis_to_entity_type(r["type_ids"]),
                    name=r["name"].lower(),
                    sources=[Source.UMLS],
                    umls_entities={"connect": [{"id": r["canonical_id"]}]},
                )
                for r in results
            ]

            # TODO: will error on duplicated names, which is exacerbated by the fact that we adjust the default UMLS name
            # e.g. "icam2", matching https://uts.nlm.nih.gov/uts/umls/concept/C1317964 and https://uts.nlm.nih.gov/uts/umls/concept/C1334075
            # also pla2g2a, fcgr3b.
            # maybe we want a 1<>many relationship between biomedical_entity and umls
            # or in this situation, we could do a WHERE on name if it fails the first time
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
            await execute()

    @staticmethod
    async def add_counts():
        """
        add counts to biomedical_entity (used for autocomplete ordering)
        TODO: counts from UMLS
        """
        logger.info("Adding biomedical counts")
        client = await prisma_client(600)
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
        """

        def get_queries(table: str) -> list[str]:
            return [
                f"DROP INDEX IF EXISTS {table}_canonical_name",  # update perf
                f"DROP INDEX IF EXISTS {table}_entity_id",  # update perf
                f"""
                UPDATE {table}
                SET
                    entity_id=entity_synonym.entity_id,
                    canonical_name=lower(biomedical_entity.name),
                    canonical_type=biomedical_entity.entity_type
                FROM entity_synonym, biomedical_entity
                WHERE {table}.name=entity_synonym.term
                AND entity_synonym.entity_id=biomedical_entity.id
                """,
            ]

        logger.info("Linking mapping tables to canonical entities")
        client = await prisma_client(600)
        for table in ["intervenable", "indicatable"]:
            for query in get_queries(table):
                await client.execute_raw(query)

        logger.warning(
            "Completed link_to_documents. ***Be sure to re-apply indices!! (prisma db push)***"
        )

    @staticmethod
    async def set_rollups():
        """
        Set instance_rollup and category_rollup for map tables (intervenable, indicatable)
        intentional denormalization for reporting (faster than recursive querying)
        """

        def get_update_queries(table: str) -> list[str]:
            return [
                # clear existing rollups
                f"DROP INDEX IF EXISTS {table}_instance_rollup",  # update perf
                f"DROP INDEX IF EXISTS {table}_category_rollup",  # update perf
                f"UPDATE {table} SET instance_rollup='', category_rollup=''",
                # logic is in how rollup_id is set (umls etl / ancestor_selector)
                f"""
                    UPDATE {table}
                    SET
                        instance_rollup=lower(instance_rollup.parent_name),
                        category_rollup=lower(category_rollup.parent_name)
                    FROM biomedical_entity AS entity
                    JOIN _rollups AS instance_rollup ON instance_rollup.child_id=entity.id
                    LEFT JOIN _rollups AS category_rollup ON category_rollup.child_id=instance_rollup.parent_id
                    WHERE {table}.entity_id=entity.id
                """,
                # ensure all records have rollups
                f"""
                    UPDATE {table} SET instance_rollup=canonical_name, category_rollup=canonical_name
                    WHERE instance_rollup=''
                """,
            ]

        # much faster with temp tables
        initialize_queries = [
            "drop table if exists _rollups",
            f"""
            -- temp table with chosen rollup parent
            CREATE TABLE _rollups AS
                SELECT
                    etp."B" as child_id,
                    max(id order by is_priority desc) as parent_id,
                    max(name order by is_priority desc) as parent_name
                FROM _entity_to_parent etp
                JOIN biomedical_entity parent_entity ON id=etp."A" -- "A" is parent
                AND parent_entity.canonical_id NOT IN {tuple(UMLS_COMMON_BASES.keys())}
                GROUP BY etp."B" -- "B" is child
            """,
            "CREATE INDEX _rollups_child_id on _rollups(child_id)",
            "CREATE INDEX _rollups_parent_id on _rollups(parent_id)",
        ]
        cleanup_queries = ["DROP TABLE IF EXISTS _rollups"]

        logger.info("Setting rollups for mapping tables")

        client = await prisma_client(600)
        for query in initialize_queries:
            await client.execute_raw(query)

        for table in ["intervenable", "indicatable"]:
            logger.info("Setting rollups for %s", table)
            update_queries = get_update_queries(table)
            for query in update_queries:
                await client.execute_raw(query)

        for query in cleanup_queries:
            await client.execute_raw(query)

        logger.warning(
            "Completed set_rollups. ***Be sure to re-apply indices!! (prisma db push)***"
        )

    @staticmethod
    async def checksum():
        """
        Quick entity checksum
        """
        client = await prisma_client(300)
        checksums = {
            "comprised_of": f"SELECT COUNT(*) FROM _entity_comprised_of",
            "parents": f"SELECT COUNT(*) FROM _entity_to_parent",
            "biomedical_entities": f"SELECT COUNT(*) FROM biomedical_entity",
            "priority_biomedical_entities": f"SELECT COUNT(*) FROM biomedical_entity where is_priority=true",
            "umls_biomedical_entities": f"SELECT COUNT(*) FROM biomedical_entity, umls where umls.id=biomedical_entity.canonical_id",
        }
        results = await asyncio.gather(
            *[client.query_raw(query) for query in checksums.values()]
        )
        for key, result in zip(checksums.keys(), results):
            logger.warning(f"Load checksum {key}: {result[0]}")
        return

    @staticmethod
    async def pre_finalize():
        """
        Finalize etl
        """
        # map umls to entities
        await BiomedicalEntityEtl._map_umls()

        # populate search index with name & syns
        await BiomedicalEntityEtl._update_search_index()

        # checksum
        await BiomedicalEntityEtl.checksum()

    @staticmethod
    async def post_finalize():
        r"""
        Run after:
            1) UMLS is loaded
            2) all biomedical entities are loaded
            3) all documents are loaded with corresponding mapping tables (intervenable, indicatable)

        Misc:
            update biomedical_entity set name=regexp_replace(name, '\ygene a\y', 'gene')
            where name ~* '\ygene a\y'
            and regexp_replace(name, '\ygene a\y', 'gene') not in
                (select name from biomedical_entity where name=regexp_replace(name, '\ygene a\y', 'gene'));
            update intervenable set canonical_name=regexp_replace(canonical_name, '\ygene a\y', 'gene') where canonical_name ~* '\ygene a\y';

        To (partly) recreate _entity_to_umls:
            UPDATE biomedical_entity set is_priority=true
                FROM umls, _entity_to_umls etu
                WHERE umls.id=etu."B"
                AND etu."A"=biomedical_entity.id
                AND (
                    ARRAY['T028', 'T116', 'T047'] && umls.type_ids
                    OR
                    biomedical_entity.name ~* '.* (?:agonists?|antagonists?|modulators?|inhibitors?)$'
                )

            INSERT into _entity_to_parent ("A", "B")
                SELECT max(parent.id order by parent.is_priority desc), child.id
                FROM biomedical_entity parent, biomedical_entity child, _entity_to_umls etu, umls
                WHERE umls.rollup_id=parent.canonical_id AND etu."A"=child.id and etu."B"=umls.id
                GROUP BY child.id
                ON CONFLICT DO NOTHING;

        Other
            select i.name, i.instance_rollup, be.name, be.entity_type, count(*) from intervenable i, biomedical_entity be where i.entity_id=be.id and be.entity_type='UNKNOWN' group by i.name, i.instance_rollup, be.entity_type, be.name order by count(*) desc limit 500;
            delete from intervenable i using biomedical_entity be, umls  where i.entity_id=be.id and be.entity_type='DISEASE' and umls.id=be.canonical_id and not umls.type_ids && ARRAY['T001', 'T004', 'T005', 'T007', 'T204'];
        """
        logger.info("Finalizing biomedical entity etl")
        await BiomedicalEntityEtl.link_to_documents()
        await BiomedicalEntityEtl.add_counts()  # TODO: counts from UMLS

        # perform final UMLS updates, which depends upon Biomedical Entities being in place.
        # NOTE: will use data/umls_ancestors.json if available, which could be stale.
        await UmlsLoader.post_doc_finalize()

        # recursively add biomedical entity parents (from UMLS)
        await BiomedicalEntityEtl._create_biomedical_entity_ancestors()

        # set instance & category rollups
        await BiomedicalEntityEtl.set_rollups()

        logger.info("Biomedical entity post_doc_finalize complete")
        return
