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

from core.ner.linker.types import CandidateSelectorType
from core.ner.normalizer import TermNormalizer
from core.ner.types import CanonicalEntity

from .types import BiomedicalEntityCreateInputWithRelationIds, RelationIdFieldMap


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
    ):
        self.normalizer = TermNormalizer(candidate_selector=candidate_selector)
        self.relation_id_field_map = relation_id_field_map

    @staticmethod
    def maybe_merge_insert_records(
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
            {
                "canonical_id": groups[0].get("canonical_id"),
                "name": groups[0]["name"],
                "entity_type": groups[0]["entity_type"],
                "synonyms": uniq(
                    compact([s for g in groups for s in g.get("synonyms") or []])
                ),
                "sources": groups[0].get("sources") or [],
                "comprised_of": uniq(
                    flatten([g.get("comprised_of") or [] for g in groups])
                ),
                "parents": uniq(flatten([g.get("parents") or [] for g in groups])),
            }
        ]

    def _generate_insert_records(
        self,
        terms_to_insert: Sequence[str],
        source_map: dict[str, dict],
        canonical_map: dict[str, CanonicalEntity],
        default_type_map: dict[str, BiomedicalEntityType] = {},
        non_canonical_source: Source = Source.FDA,
        synonym_fields: list[str] = ["brand_name"],
    ) -> list[BiomedicalEntityCreateInputWithRelationIds]:
        """
        Create records for entity insert
        """

        def get_insert_record(
            orig_name: str,
        ) -> BiomedicalEntityCreateInputWithRelationIds:
            source_rec = source_map.get(orig_name)
            canonical = canonical_map.get(orig_name)

            if source_rec is not None:
                rel_fields: dict[str, list[str]] = {
                    rel_field: uniq(
                        compact(
                            [
                                canonical_map[i].id if i in canonical_map else None
                                for i in source_rec[source_field]
                            ]
                        )
                    )
                    for rel_field, source_field in self.relation_id_field_map.items()
                }
                source_dependent_fields = {
                    "synonyms": [
                        orig_name,
                        *[str(source_rec[sf]) for sf in synonym_fields],
                    ],
                    **rel_fields,
                }
            else:
                source_dependent_fields = {
                    "synonyms": [orig_name],
                }

            if canonical is not None:
                canonical_dependent_fields = {
                    "canonical_id": canonical.id,
                    "name": canonical.name.lower(),
                    "entity_type": canonical.type,
                    "sources": [Source.UMLS],
                }
            else:
                canonical_dependent_fields = {
                    "canonical_id": None,
                    "name": orig_name,
                    "entity_type": default_type_map.get(orig_name)
                    or BiomedicalEntityType.UNKNOWN,
                    "sources": [non_canonical_source],
                }

            return BiomedicalEntityCreateInputWithRelationIds(
                **{
                    **canonical_dependent_fields,  # type: ignore
                    **source_dependent_fields,
                }
            )

        # merge records with same canonical id
        def merge_records():
            flat_recs = [get_insert_record(name) for name in terms_to_insert]
            grouped_recs = group_by(flat_recs, "canonical_id")
            merged_recs = flatten(
                [
                    BiomedicalEntityEtl.maybe_merge_insert_records(groups, cid)
                    for cid, groups in grouped_recs.items()
                ]
            )

            return merged_recs

        insert_records = merge_records()
        return insert_records

    def generate_canonical_map(
        self, terms_to_canonicalize: Sequence[str]
    ) -> dict[str, CanonicalEntity]:
        """
        Generate canonical map for source terms
        """

        # normalize all intervention names, except if combos
        canonical_docs = self.normalizer.normalize_strings(terms_to_canonicalize)

        # map for quick lookup of canonical entities
        canonical_map = {
            on: de.canonical_entity
            for on, de in zip(terms_to_canonicalize, canonical_docs)
            if de.canonical_entity is not None
        }
        return canonical_map

    async def create_records(
        self,
        terms_to_canonicalize: Sequence[str],
        terms_to_insert: Sequence[str],
        source_map: dict,
        default_type_map: dict[str, BiomedicalEntityType] = {},
    ):
        """
        Create records for entities and relationships

        Args:
            terms_to_canonicalize (Sequence[str]): terms to canonicalize
            terms_to_insert (Sequence[str]): terms to insert
            source_map (dict): map of "original_term" to source record
            default_type_map (dict): map of terms to default entity types
        """
        canonical_map = self.generate_canonical_map(terms_to_canonicalize)
        entity_recs = self._generate_insert_records(
            terms_to_insert,
            source_map,
            canonical_map,
            default_type_map=default_type_map,
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
        recs_with_relations = [
            er
            for er in entity_recs
            if (any([er.get(k) is not None for k in self.relation_id_field_map.keys()]))
            is not None
        ]
        for rwr in recs_with_relations:
            update = BiomedicalEntityUpdateInput(
                **{  # type: ignore
                    k: {"connect": [{"canonical_id": co} for co in rwr.get(k) or []]}  # type: ignore
                    for k in self.relation_id_field_map.keys()
                },
            )
            await BiomedicalEntity.prisma().update(
                where={"name": rwr["name"]}, data=update
            )
