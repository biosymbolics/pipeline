from dataclasses import dataclass
import re
from typing import Literal

from utils.classes import ByDefinitionOrderEnum
from utils.re import get_or_re

GENERAL_CATEGORY_TYPE_IDS = {
    "T170": "Intellectual Product",  # e.g. MeSH Russian
    "T185": "Classification",  # e.g. Genes and Gene Therapy, Schedule I Substance, Species
    "T091": "Biomedical Occupation or Discipline",
}

GENERAL_CATEGORY_IDS = {
    "C1328819": "Small Molecule",
    "C1258062": "Molecular Mechanisms of Pharmacological Action",
    "C0600511": "Pharmacologic Actions",
    "C1258064": "Therapeutic Uses",
}

GENERAL_CATEGORY_RES = [
    "categor(?:ized|y|ies)",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C0729758
    "and",
    "or",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C0729473
    "class",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C1040746
    "unclassified",  # https://uts.nlm.nih.gov/uts/umls/concept/C1621543
    "genus",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C0004587
    "miscellaneous",  # https://uts.nlm.nih.gov/uts/umls/concept/C1513329
    "phenomena",  # https://uts.nlm.nih.gov/uts/umls/concept/C2350469
    "other",  # https://uts.nlm.nih.gov/uts/umls/concept/C3653837
]

GENERAL_CATEGORY_RE = get_or_re(
    GENERAL_CATEGORY_RES, enforce_word_boundaries=True, permit_plural=False
)

INSTANCE_TYPE_IDS = {
    "T200": "Clinical Drug",  # e.g. https://uts.nlm.nih.gov/uts/umls/concept/C3531481
}


@dataclass(frozen=True)
class UmlsRecord:
    def __getitem__(self, item):
        return getattr(self, item)

    id: str
    canonical_name: str
    num_descendants: int
    hierarchy: str | None
    type_id: str
    type_name: str


def is_name_semantic_type(canonical_name: str, type_name: str) -> bool:
    """
    True if canonical_name is also a semantic type name, or close, e.g.
    - Enzymes vs Enzyme (type is typically singular)
    - TODO: Fungi vs Fungus (https://uts.nlm.nih.gov/uts/umls/concept/C0016832)
    - Biologically Active Substance (https://uts.nlm.nih.gov/uts/umls/concept/C0574031)
    """
    return (
        re.search(
            re.escape(canonical_name) + "?",
            type_name,
            re.IGNORECASE,
        )
        is not None
    )


THRESHOLD_COUNT_FOR_CATEGORY = 200
THRESHOLD_COUNT_FOR_INSTANCE = 5


class OntologyLevel(ByDefinitionOrderEnum):
    GENERAL_CATEGORY = "GENERAL_CATEGORY"  # least specific
    CATEGORY = "CATEGORY"
    INSTANCE = "INSTANCE"  # most specific
    UNKNOWN = "UNKNOWN"

    @classmethod
    def find_by_record(cls, record: UmlsRecord, ancestor_ids: list[str]):
        return cls.find(
            record["canonical_name"],
            record["type_id"],
            record["type_name"],
            record["num_descendants"],
            ancestor_ids,
        )

    @classmethod
    def find(
        cls,
        canonical_name: str,
        type_id: str,
        type_name: str,
        num_descendants: int,
        ancestor_ids: list[str],
    ):
        """
        Simple heuristic to find approximate semantic level of UMLS record
        TODO: make this a model
        """
        if (
            (len(ancestor_ids) == 0 and num_descendants > 0)
            or type_id in GENERAL_CATEGORY_TYPE_IDS.keys()
            or re.search(GENERAL_CATEGORY_RE, canonical_name) is not None
            or is_name_semantic_type(canonical_name, type_name)
            or num_descendants >= THRESHOLD_COUNT_FOR_CATEGORY
        ):
            return cls.GENERAL_CATEGORY

        if (
            num_descendants <= THRESHOLD_COUNT_FOR_INSTANCE
            or type_id in INSTANCE_TYPE_IDS.keys()
        ):
            return cls.INSTANCE

        return cls.CATEGORY


RollupLevel = Literal[OntologyLevel.INSTANCE, OntologyLevel.INSTANCE]  # type: ignore


@dataclass(frozen=True)
class UmlsLookupRecord(UmlsRecord):
    level: OntologyLevel
    instance_rollup: str | None
    category_rollup: str | None
