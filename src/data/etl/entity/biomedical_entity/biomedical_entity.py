"""
Class for biomedical entity etl
"""
from typing import Sequence
from prisma.models import BiomedicalEntity
from prisma.enums import BiomedicalEntityType, Source
from prisma.types import (
    BiomedicalEntityCreateInput,
    BiomedicalEntityUpdateInput,
    BiomedicalEntityUpsertInput,
)
from pydash import flatten, group_by

from clients.low_level.prisma import batch_update, prisma_client
from constants.umls import UMLS_DISEASE_TYPES, UMLS_TARGET_TYPES
from core.ner.cleaning import CleanFunction
from core.ner.linker.types import CandidateSelectorType
from core.ner.normalizer import TermNormalizer
from core.ner.types import CanonicalEntity
from data.domain.biomedical.umls import tuis_to_entity_type
from data.etl.types import RelationIdFieldMap
from typings.documents.common import ENTITY_MAP_TABLES
from utils.list import merge_nested

from .umls.load import UmlsLoader
from ..base_entity_etl import BaseEntityEtl

DEFAULT_TYPE_FIELD = "default_type"
OVERRIDE_TYPE_FIELD = "type"


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

    def _generate_lookup_map(self, terms: Sequence[str]) -> dict[str, CanonicalEntity]:
        """
        Generate canonical map for source terms
        """

        lookup_docs = self.normalizer.normalize_strings(terms)

        # map for quick lookup of canonical entities
        lookup_map = {
            on: de.canonical_entity
            for on, de in zip(terms, lookup_docs)
            if de.canonical_entity is not None
        }
        return lookup_map

    def _generate_upsert_records(
        self,
        terms_to_insert: Sequence[str],
        source_map: dict[str, dict],
        canonical_map: dict[str, CanonicalEntity],
    ) -> list[BiomedicalEntityCreateInput]:
        """
        Create record dicts for entity insert
        """

        def create_input(orig_name: str) -> BiomedicalEntityCreateInput:
            """
            Form create input for a given term
            """
            source_rec = source_map[orig_name]
            canonical = canonical_map.get(orig_name)

            # create input for N-to-N relationships (synonyms, comprised_of, parents)
            # taken from source map
            relation_fields = {
                field: rel_spec.form_prisma_relation(source_rec, canonical_map)
                for field, rel_spec in self.relation_id_field_map.items()
                if rel_spec is not None
            }

            if canonical is not None:
                canonical_dependent_fields = BiomedicalEntityCreateInput(
                    canonical_id=canonical.id or canonical.name.lower(),
                    entity_type=source_rec.get(OVERRIDE_TYPE_FIELD) or canonical.type,
                    is_priority=source_rec.get("is_priority") or False,
                    name=canonical.name.lower(),
                    sources=[Source.UMLS],
                    # An operation failed because it depends on one or more records that were required but not found.
                    # umls_entities={
                    #     "connect": [{"id": id} for id in canonical.ids or []]
                    # },
                )
            else:
                canonical_dependent_fields = BiomedicalEntityCreateInput(
                    canonical_id=orig_name,
                    entity_type=(
                        source_rec.get(DEFAULT_TYPE_FIELD)
                        or BiomedicalEntityType.UNKNOWN
                    ),
                    is_priority=source_rec.get("is_priority") or False,
                    name=orig_name,
                    sources=[self.non_canonical_source],
                )

            return BiomedicalEntityCreateInput(
                **canonical_dependent_fields,
                **relation_fields,
            )

        # merge records with same canonical id
        flat_recs = [create_input(name) for name in terms_to_insert]
        grouped_recs = group_by(flat_recs, "canonical_id")
        upsert_records = flatten(
            [
                [BiomedicalEntityCreateInput(**merge_nested(*groups))]  # type: ignore
                if canonical_id is None
                else groups
                for canonical_id, groups in grouped_recs.items()
            ]
        )

        return upsert_records

    async def copy_all(
        self,
        terms: Sequence[str],
        terms_to_canonicalize: Sequence[str],
        source_map: dict[str, dict],
    ):
        """
        Create records for entities and relationships

        Args:
            terms (Sequence[str]): terms to insert
            terms_to_canonicalize (Sequence[str]): terms to canonicalize, if different than terms
            source_map (dict): map of "term" to source record for additional fields, e.g. synonyms, "active_ingredients", etc.
        """
        canonical_map = self._generate_lookup_map(terms_to_canonicalize or terms)
        entity_recs = self._generate_upsert_records(terms, source_map, canonical_map)

        await batch_update(
            entity_recs,
            update_func=lambda r, tx: BiomedicalEntity.prisma(tx).upsert(
                data=BiomedicalEntityUpsertInput(
                    create=r,
                    update=BiomedicalEntityUpdateInput(**r),  # type: ignore
                ),
                where={"canonical_id": r["canonical_id"]},
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
    async def _umls_to_biomedical_entity(n_depth: int = 3):
        """
        Create biomedical entity records for UMLS parents.

        Args:
            n_depth (int): numer of iterations / recursion depth
                e.g. iteration 1, we get parents. iteration 2, we get parents of parents, etc.
        """

        async def execute():
            query = """
                SELECT
                    biomedical_entity.id AS child_id,
                    umls_parent.id AS canonical_id,
                    umls_parent.preferred_name AS name,
                    umls_parent.type_ids AS tuis
                FROM biomedical_entity, umls, _entity_to_umls etu, umls_parent
                WHERE
                    etu."A"=biomedical_entity.id
                    AND umls.id=etu."B"
                    AND umls_parent.id=umls.rollup_id
            """
            client = await prisma_client(300)
            results = await client.query_raw(query)
            records = [
                BiomedicalEntityCreateInput(
                    canonical_id=r["canonical_id"],
                    # # a problem in a txn?
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
                    where={"canonical_id": r["canonical_id"]},
                ),
                batch_size=1000,
            )

        for _ in range(n_depth):
            await execute()

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

        TODO: add UMLS to biomedical_entity hierarchy; select rollups from there
            (which will include the MoAs we add in approvals load)
            Also, remember to use approval MoAs as nucleators.
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
    async def add_rollups():
        """
        Set instance_rollup and category_rollup for map tables (intervenable, indicatable)
        intentional denormalization for reporting (faster than querying biomedical_entity or umls)

        TODO: look only at biomedical_entity & parent/child relationships
        """

        def get_query(table: str, filters: list[str] = []) -> str:
            return f"""
                UPDATE {table}
                SET
                    category_rollup=lower(umls_category_rollup.preferred_name),
                    instance_rollup=lower(umls_instance_rollup.preferred_name)
                FROM biomedical_entity, _entity_to_umls as etu, umls
                JOIN umls as umls_rollup on umls_rollup.id=umls.rollup_id
                LEFT JOIN umls as umls_category_rollup on umls_category_rollup.id=umls_instance_rollup.rollup_id
                WHERE {table}.entity_id=biomedical_entity.id
                AND biomedical_entity.id=etu."A"
                AND umls.id=etu."B"
                {'AND ' + ' AND '.join(filters) if filters else ''}
            """

        # specification for linking docs to ents & umls
        spec = {
            "intervenable": [
                {
                    # prefer target types (protein/gene)
                    "filters": [
                        f"umls.type_ids && ARRAY{list(UMLS_TARGET_TYPES.keys())}"
                    ],
                },
                {
                    # then, only update those not yet updated
                    "filters": ["instance_rollup = ''"],
                },
            ],
            "indicatable": [
                {
                    # prefer disease types
                    "filters": [
                        f"umls.type_ids && ARRAY{list(UMLS_DISEASE_TYPES.keys())}"
                    ],
                },
                {
                    # then, only update those not yet updated
                    "filters": ["instance_rollup = ''"],
                },
            ],
        }

        # execute the spec
        client = await prisma_client(600)
        for table, specs in spec.items():
            for spec in specs:
                query = get_query(table, spec["filters"])
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
        await BiomedicalEntityEtl.link_to_documents()
        await BiomedicalEntityEtl.add_counts()

        # perform final UMLS updates, which depends upon Biomedical Entities being in place.
        await UmlsLoader.update_with_ontology_level()

        # recursively add parents from UMLS
        await BiomedicalEntityEtl._umls_to_biomedical_entity()

        # add instance & category rollups
        # TODO: should get rollups via biomedical_entity
        await BiomedicalEntityEtl.add_rollups()
