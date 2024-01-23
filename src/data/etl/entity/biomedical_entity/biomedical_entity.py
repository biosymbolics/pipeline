"""
Class for biomedical entity etl
"""
from typing import Sequence
from prisma.models import BiomedicalEntity
from prisma.enums import BiomedicalEntityType, Source
from prisma.types import (
    BiomedicalEntityUpdateInput,
    BiomedicalEntityCreateWithoutRelationsInput,
)
from pydash import compact, flatten, group_by, omit, uniq

from clients.low_level.prisma import prisma_client
from constants.umls import UMLS_DISEASE_TYPES, UMLS_TARGET_TYPES
from core.ner.cleaning import CleanFunction
from core.ner.linker.types import CandidateSelectorType
from core.ner.normalizer import TermNormalizer
from core.ner.types import CanonicalEntity
from data.etl.types import (
    BiomedicalEntityCreateInputWithRelationIds as BiomedicalEntityCreateInput,
    RelationIdFieldMap,
)
from typings.documents.common import ENTITY_MAP_TABLES
from utils.file import save_json_as_file

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

    def _maybe_merge_insert_records(
        self,
        groups: list[BiomedicalEntityCreateInput],
        canonical_id: str,
    ) -> list[BiomedicalEntityCreateInput]:
        """
        Merge records with same canonical id
        """
        # if no canonical id, then no merging
        if canonical_id is None:
            return groups

        return [
            BiomedicalEntityCreateInput(
                canonical_id=groups[0].get("canonical_id"),
                name=groups[0]["name"],
                entity_type=groups[0]["entity_type"],
                sources=groups[0].get("sources") or [],
                # add in relationship ids to add in subsequent update
                **{
                    k: uniq(flatten([g.get(k) or [] for g in groups]))
                    for k in self.relation_id_field_map.keys()
                },  # type: ignore
            )
        ]

    def _generate_insert_records(
        self,
        terms_to_insert: Sequence[str],
        source_map: dict[str, dict],
        canonical_map: dict[str, CanonicalEntity],
    ) -> list[BiomedicalEntityCreateInput]:
        """
        Create record dicts for entity insert
        """

        def create_input(
            orig_name: str,
        ) -> BiomedicalEntityCreateInput:
            """
            Form create input for a given term
            """
            source_rec = source_map[orig_name]
            canonical = canonical_map.get(orig_name)

            # fields for N-to-N relationships (synonyms, comprised_of, parents)
            relation_fields: dict[str, list[str]] = {
                rel_field: uniq(
                    compact(
                        [
                            connect_info.get_value(val, canonical_map)
                            for val in source_rec.get(connect_info.source_field) or []
                        ]
                    )
                )
                for rel_field, connect_info in self.relation_id_field_map.items()
                if connect_info is not None
            }

            if canonical is not None:
                entity_type = source_rec.get(OVERRIDE_TYPE_FIELD) or canonical.type
                canonical_dependent_fields = {
                    "canonical_id": canonical.id,
                    "name": canonical.name.lower(),
                    "entity_type": entity_type,
                    "sources": [Source.UMLS],
                    "umls": {
                        "connect": {"id": {"in": canonical.ids}},
                    },
                }
            else:
                entity_type = (
                    source_rec.get(DEFAULT_TYPE_FIELD) or BiomedicalEntityType.UNKNOWN
                )
                canonical_dependent_fields = {
                    "canonical_id": None,
                    "name": orig_name,
                    "entity_type": entity_type,
                    "sources": [self.non_canonical_source],
                }

            return BiomedicalEntityCreateInput(
                **{
                    **canonical_dependent_fields,  # type: ignore
                    **relation_fields,
                }
            )

        # merge records with same canonical id
        def merge_records():
            flat_recs = [create_input(name) for name in terms_to_insert]
            grouped_recs = group_by(flat_recs, "canonical_id")
            merged_recs = flatten(
                [
                    self._maybe_merge_insert_records(groups, cid)
                    for cid, groups in grouped_recs.items()
                ]
            )

            return merged_recs

        insert_records = merge_records()
        save_json_as_file(insert_records, "insert_records.json")
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

        # create flat records
        await BiomedicalEntity.prisma().create_many(
            data=[
                BiomedicalEntityCreateWithoutRelationsInput(
                    **omit(er, *self.relation_id_field_map.keys())  # type: ignore
                )
                for er in entity_recs
            ],
            skip_duplicates=True,
        )

        # update records with relationships (parents, comprised_of, synonyms)
        for entity_rec in entity_recs:
            update = BiomedicalEntityUpdateInput(
                **{
                    k: connect_info.form_prisma_relation(entity_rec)
                    for k, connect_info in self.relation_id_field_map.items()
                    if connect_info is not None
                },  # type: ignore
            )
            try:
                await BiomedicalEntity.prisma().update(
                    where={"name": entity_rec["name"]}, data=update
                )
            except Exception as e:
                print(e, entity_rec)

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
                from synonym where entity_id=biomedical_entity.id;
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
            insert into _entity_to_umls ("A", "B")
            select s.id, cid from (
                select id, unnest(string_to_array(canonical_id, '|')) as cid
                from biomedical_entity
                where canonical_id is not null
            ) s
            JOIN umls on umls.id = s.cid
            on conflict do nothing;
        """
        client = await prisma_client(600)
        await client.execute_raw(query)

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
        Set instance_rollup and category_rollup for biomedical entities

        TODO: add UMLS to biomedical_entity hierarchy; select rollups from there
            (which will include the MoAs we add in approvals load)
            Also, remember to use approval MoAs as nucleators.
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
                    # prefer target linkage (protein/gene)
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
                    # prefer disease linkage
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
