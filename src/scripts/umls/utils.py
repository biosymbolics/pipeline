from typing import Sequence
import logging

from data.common.biomedical.umls import clean_umls_name
from typings.umls import (
    IntermediateUmlsRecord,
    OntologyLevel,
    UmlsLookupRecord,
    UmlsRecord,
)

from .ancestor_selection import UmlsGraph
from .constants import MAX_DENORMALIZED_ANCESTORS
from .types import is_umls_record_list


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def find_level_ancestor(
    record: IntermediateUmlsRecord,
    level: OntologyLevel,
    ancestors: tuple[IntermediateUmlsRecord, ...],
) -> str:
    """
    Find first ancestor at the specified level

    Args:
        record (UmlsLookupRecord): UMLS record
        level (OntologyLevel): level to find
        ancestors (tuple[UmlsLookupRecord]): ordered list of ancestors

    Returns (str): ancestor id, or "" if none found
    """
    level_ancestors = [a for a in ancestors if a["level"] == level]

    if len(level_ancestors) == 0:
        # return self as instance ancestor if no ancestors
        if record["level"] == level:
            return record["id"]

        return ""

    # for instance level, use the last "INSTANCE" ancestor
    if level == OntologyLevel.INSTANCE:
        return level_ancestors[-1]["id"]

    # else, use self as ancestor if matching level
    if record["level"] == level:
        return record["id"]

    # otherwise, use the first matching ancestor
    return level_ancestors[0]["id"]


class UmlsTransformer:
    """
    Class for transforming UMLS records
    """

    def __init__(self, aui_lookup: dict[str, str]):
        self.aui_lookup: dict[str, str] = aui_lookup
        self.lookup_dict: dict[str, IntermediateUmlsRecord] | None = None
        self.betweenness_map: dict[str, float] = UmlsGraph().betweenness_map

    def initialize(self, all_records: Sequence[dict]):
        if not is_umls_record_list(all_records):
            raise ValueError(f"Records are not UmlsRecords: {all_records[:10]}")

        logger.info("Initializing UMLS lookup dict with %s records", len(all_records))
        lookup_records = [self._create_lookup_record(r) for r in all_records]
        self.lookup_dict = {r["id"]: r for r in lookup_records}
        logger.info("Initializing UMLS lookup dict")

    def _create_lookup_record(self, record: UmlsRecord) -> IntermediateUmlsRecord:
        """
        Create a record for each UMLS entity, to be used for internal purposes
        """
        hier = record["hierarchy"]
        # reverse to get nearest ancestor first
        ancestors = (hier.split(".") if hier is not None else [])[::-1]
        ancestor_cuis = [self.aui_lookup.get(aui, "") for aui in ancestors]

        return {
            **record,  # type: ignore
            **{f"l{i}_ancestor": None for i in range(MAX_DENORMALIZED_ANCESTORS)},
            **{
                f"l{i}_ancestor": ancestor_cuis[i] if i < len(ancestor_cuis) else None
                for i in range(MAX_DENORMALIZED_ANCESTORS)
            },
            "level": OntologyLevel.find(record["id"], self.betweenness_map),
            "preferred_name": clean_umls_name(
                record["id"], record["canonical_name"], record["synonyms"], False
            ),
        }

    def transform_record(self, r: IntermediateUmlsRecord) -> UmlsLookupRecord:
        """
        Transform a single UMLS record, intended to be persisted
        """
        if self.lookup_dict is None:
            raise ValueError("Lookup dict is not initialized")

        cuis = [r[f"l{i}_ancestor"] for i in range(MAX_DENORMALIZED_ANCESTORS)]
        ancestors = tuple(
            [self.lookup_dict[cui] for cui in cuis if cui in self.lookup_dict]
        )
        return UmlsLookupRecord.from_intermediate(
            r,
            instance_rollup=find_level_ancestor(r, OntologyLevel.INSTANCE, ancestors),
            category_rollup=find_level_ancestor(
                r, OntologyLevel.L1_CATEGORY, ancestors
            ),
        )

    def __call__(
        self,
        batch: Sequence[dict],
        all_records: Sequence[dict],
    ) -> list[UmlsLookupRecord]:
        """
        Transform umls relationship

        Args:
            batch (Sequence[dict]): batch of records to transform
            all_records (Sequence[dict]): all UMLs records
        """
        if not is_umls_record_list(batch):
            raise ValueError(f"Records are not UmlsRecords: {batch[:10]}")

        if self.lookup_dict is None:
            self.initialize(all_records)
            assert self.lookup_dict is not None

        batch_records = [self.lookup_dict[r["id"]] for r in batch]

        return [self.transform_record(r) for r in batch_records]
