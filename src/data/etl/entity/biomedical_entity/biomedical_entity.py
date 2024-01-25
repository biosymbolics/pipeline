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
from pydash import flatten, group_by, uniq

from clients.low_level.prisma import batch_update, prisma_client
from constants.umls import UMLS_DISEASE_TYPES, UMLS_TARGET_TYPES
from core.ner.cleaning import CleanFunction
from core.ner.linker.types import CandidateSelectorType
from core.ner.normalizer import TermNormalizer
from core.ner.types import CanonicalEntity
from data.domain.biomedical.umls import tuis_to_entity_type
from data.etl.types import RelationIdFieldMap
from typings.documents.common import ENTITY_MAP_TABLES

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

    def _generate_insert_records(
        self,
        terms_to_insert: Sequence[str],
        source_map: dict[str, dict],
        canonical_map: dict[str, CanonicalEntity],
    ) -> list[BiomedicalEntityUpdateInput]:
        """
        Create record dicts for entity insert
        """

        def create_input(orig_name: str) -> BiomedicalEntityUpdateInput:
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
                entity_type = source_rec.get(OVERRIDE_TYPE_FIELD) or canonical.type
                canonical_dependent_fields = {
                    "canonical_id": canonical.id,
                    "entity_type": entity_type,
                    "is_preferred": source_rec.get("is_preferred") or False,
                    "name": canonical.name.lower(),
                    "sources": [Source.UMLS],
                    "umls": {"connect": {"id": {"in": canonical.ids}}},
                }
            else:
                entity_type = (
                    source_rec.get(DEFAULT_TYPE_FIELD) or BiomedicalEntityType.UNKNOWN
                )
                canonical_dependent_fields = {
                    "canonical_id": None,
                    "entity_type": entity_type,
                    "is_preferred": source_rec.get("is_preferred") or False,
                    "name": orig_name,
                    "sources": [self.non_canonical_source],
                }

            return BiomedicalEntityUpdateInput(
                **canonical_dependent_fields,  # type: ignore
                **relation_fields,
            )

        def _maybe_merge_insert_records(
            groups: list[BiomedicalEntityUpdateInput],
            canonical_id: str,
        ) -> list[BiomedicalEntityUpdateInput]:
            """
            Merge records with same canonical id
            """
            # if no canonical id, then no merge. return all.
            if canonical_id is None:
                return groups

            # merge relationship ids across groups
            merged = {
                **groups[0].__dict__,
                **{
                    field: uniq(flatten([g.get(field) or [] for g in groups]))
                    for field in self.relation_id_field_map.keys()
                },
            }

            return [BiomedicalEntityUpdateInput(**merged)]

        # merge records with same canonical id
        flat_recs = [create_input(name) for name in terms_to_insert]
        grouped_recs = group_by(flat_recs, "canonical_id")
        insert_records = flatten(
            [
                _maybe_merge_insert_records(groups, canonical_id)
                for canonical_id, groups in grouped_recs.items()
            ]
        )

        return insert_records

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
            source_map (dict): map of "original_term" to source record
                               for additional fields, e.g. synonyms, "active_ingredients", etc.
            terms_to_canonicalize (Sequence[str]): terms to canonicalize, if different than terms
        """
        canonical_map = self._generate_lookup_map(terms_to_canonicalize or terms)
        entity_recs = self._generate_insert_records(
            terms,
            source_map,
            canonical_map,
        )

        await batch_update(
            entity_recs,
            update_func=lambda r, tx: BiomedicalEntity.prisma(tx).upsert(
                data=BiomedicalEntityUpsertInput(
                    create=BiomedicalEntityCreateInput(**r),  # type: ignore
                    update=BiomedicalEntityUpdateInput(**r),
                ),
                where={"name": r["name"]},  # type: ignore
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
        Map UMLS to entities, based on id field (which we know to be a pipe-delimited list of UMLS ids (+random strings we ignore)

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
    async def _umls_to_biomedical_entity():
        """
        Create biomedical entity records for UMLS parents.
        """
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
                AND umls_parent.id=umls.instance_rollup_id -- category_rollup_id next?
        """
        client = await prisma_client(300)
        results = await client.query_raw(query)
        records = [
            {
                "canonical_id": r["canonical_id"],
                "children": {"connect": {"id": r["child_id"]}},  # a problem in a txn?
                "entity_type": tuis_to_entity_type(r["tuis"]),
                "name": r["name"],
                "sources": [Source.UMLS],
                "umls": {"connect": {"id": {"equals": r["canonical_id"]}}},
            }
            for r in results
        ]

        await batch_update(
            records,
            update_func=lambda r, tx: BiomedicalEntity.prisma(tx).upsert(
                data={
                    "create": BiomedicalEntityCreateInput(**r),  # type: ignore
                    "update": BiomedicalEntityUpdateInput(**r),  # type: ignore
                },
                where={"canonical_id": r["canonical_id"]},  # type: ignore
            ),
            batch_size=1000,
        )

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
        """

        def get_query(table: str, filters: list[str] = []) -> str:
            return f"""
                UPDATE {table}
                SET
                    category_rollup=lower(umls_category_rollup.preferred_name),
                    instance_rollup=lower(umls_instance_rollup.preferred_name)
                FROM biomedical_entity,
                    _entity_to_umls as etu,
                    umls as umls_instance_rollup,
                    umls
                LEFT JOIN umls as umls_category_rollup on umls_category_rollup.id=umls.category_rollup_id
                WHERE {table}.entity_id=biomedical_entity.id
                AND biomedical_entity.id=etu."A"
                AND umls.id=etu."B"
                AND umls_instance_rollup.id=umls.instance_rollup_id
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

        # perform final Umls updates, which depends upon Biomedical Entities being in place.
        await UmlsLoader.update_with_ontology_level()

        # add instance & category rollups
        await BiomedicalEntityEtl.add_rollups()
