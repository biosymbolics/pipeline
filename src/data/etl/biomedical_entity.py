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

from core.ner.cleaning import CleanFunction
from core.ner.linker.types import CandidateSelectorType
from core.ner.normalizer import TermNormalizer
from core.ner.types import CanonicalEntity

from .types import (
    BiomedicalEntityCreateInputWithRelationIds,
    RelationIdFieldMap,
)

DEFAULT_TYPE_FIELD = "default_type"


class BiomedicalEntityEtl:
    """
    Class for biomedical entity etl

    - canonicalizes/normalizes terms
    - creates records for entities and corresponding relationships (e.g. parents, comprised_of)
    """

    def __init__(
        self,
        candidate_selector: CandidateSelectorType,
        relation_id_field_map: RelationIdFieldMap,
        additional_cleaners: Sequence[CleanFunction] = [],
    ):
        self.normalizer = TermNormalizer(
            candidate_selector=candidate_selector,
            additional_cleaners=additional_cleaners,
        )
        self.relation_id_field_map = relation_id_field_map

    def maybe_merge_insert_records(
        self,
        groups: list[BiomedicalEntityCreateInputWithRelationIds],
        canonical_id: str,
    ) -> list[BiomedicalEntityCreateInputWithRelationIds]:
        """
        Merge records with same canonical id
        """
        # if no canonical id, then no merging
        if canonical_id is None:
            return groups

        return [
            BiomedicalEntityCreateInputWithRelationIds(
                **{
                    "canonical_id": groups[0].get("canonical_id"),
                    "name": groups[0]["name"],
                    "entity_type": groups[0]["entity_type"],
                    "sources": groups[0].get("sources") or [],
                    **{
                        k: uniq(flatten([g.get(k) or [] for g in groups]))
                        for k in self.relation_id_field_map.keys()
                    },  # type: ignore
                }
            )
        ]

    def _generate_insert_records(
        self,
        terms_to_insert: Sequence[str],
        source_map: dict[str, dict],
        canonical_map: dict[str, CanonicalEntity],
        non_canonical_source: Source = Source.FDA,
    ) -> list[BiomedicalEntityCreateInputWithRelationIds]:
        """
        Create record dicts for entity insert
        """

        def get_insert_record(
            orig_name: str,
        ) -> BiomedicalEntityCreateInputWithRelationIds:
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
                canonical_dependent_fields = {
                    "canonical_id": canonical.id,
                    "name": canonical.name.lower(),
                    "entity_type": canonical.type,
                    "sources": [Source.UMLS],
                }
            else:
                entity_type = (
                    source_rec.get(DEFAULT_TYPE_FIELD) or BiomedicalEntityType.UNKNOWN
                )
                canonical_dependent_fields = {
                    "canonical_id": None,
                    "name": orig_name,
                    "entity_type": entity_type,
                    "sources": [non_canonical_source],
                }

            return BiomedicalEntityCreateInputWithRelationIds(
                **{
                    **canonical_dependent_fields,  # type: ignore
                    **relation_fields,
                }
            )

        # merge records with same canonical id
        def merge_records():
            flat_recs = [get_insert_record(name) for name in terms_to_insert]
            grouped_recs = group_by(flat_recs, "canonical_id")
            merged_recs = flatten(
                [
                    self.maybe_merge_insert_records(groups, cid)
                    for cid, groups in grouped_recs.items()
                ]
            )

            return merged_recs

        insert_records = merge_records()
        return insert_records

    def generate_lookup_map(self, terms: Sequence[str]) -> dict[str, CanonicalEntity]:
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

    async def create_records(
        self,
        terms_to_canonicalize: Sequence[str],
        terms_to_insert: Sequence[str],
        source_map: dict[str, dict],
    ):
        """
        Create records for entities and relationships

        Args:
            terms_to_canonicalize (Sequence[str]): terms to canonicalize
            terms_to_insert (Sequence[str]): terms to insert
            source_map (dict): map of "original_term" to source record
                               for additional fields, e.g. synonyms, "active_ingredients", etc.
        """
        canonical_map = self.generate_lookup_map(terms_to_canonicalize)
        entity_recs = self._generate_insert_records(
            terms_to_insert,
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

        # update records with relationships with connection info
        for entity_rec in entity_recs:
            update = BiomedicalEntityUpdateInput(
                **{
                    k: connect_info.form_prisma_relation(entity_rec)
                    for k, connect_info in self.relation_id_field_map.items()
                    if connect_info is not None
                },  # type: ignore
            )
            await BiomedicalEntity.prisma().update(
                where={"name": entity_rec["name"]}, data=update
            )