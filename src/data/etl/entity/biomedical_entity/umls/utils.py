from functools import reduce
import math
from typing import Callable, Mapping, Sequence
from pydash import compact, flatten, uniq
from prisma.enums import OntologyLevel
import logging

from constants.umls import PREFERRED_ANCESTOR_TYPE_MAP
from data.etl.entity.biomedical_entity.umls.constants import ONTOLOGY_LEVEL_MAP

from .types import UmlsInfo

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def choose_best_ancestor(
    child: UmlsInfo, ancestors: Sequence[UmlsInfo]
) -> UmlsInfo | None:
    """
    Choose the best available ancestor

    Args:
        child_types (list[str]): list of child types
        ancestors (list[str]): list of ancestors
    """
    score_composite = get_composite_ancestor_scorer(child)

    best_ancestors = sorted(
        [a for a in ancestors if score_composite(a) > 0],
        key=lambda a: score_composite(a),
        reverse=True,
    )

    if len(best_ancestors) == 0:
        return None

    return best_ancestors[0]


def get_composite_ancestor_scorer(child: UmlsInfo) -> Callable[[UmlsInfo], int]:
    """
    Returns a composite ancestor scorer function
    (higher is better; -1 means disqualified)

    Args:
        child (UmlsInfo): child

    Returns (Callable[[UmlsInfo], int]): a function that scores an ancestor by type and level
    """
    score_ancestor_by_type = get_ancestor_type_scorer(child.type_ids)
    score_ancestor_by_level = get_ancestor_level_scorer(child.level)

    def score_composite(a: UmlsInfo) -> int:
        type_score = score_ancestor_by_type(a.type_ids)
        level_score = score_ancestor_by_level(a.level)

        if type_score < 0 or level_score < 0:
            return -1

        # higher is better (but not that much better, thus log)
        return round(math.log(type_score) + math.log(level_score))

    return score_composite


def get_ancestor_level_scorer(
    child_level: OntologyLevel,
) -> Callable[[OntologyLevel], int]:
    """
    Returns an ancestor level scorer function

    Args:
        child_level (OntologyLevel): child level

    Returns (Callable[[OntologyLevel], int]): a function that scores an ancestor by level
        higher is better; -1 means disqualified
    """
    MAX_ONTOLOGY_SCORE = max(ONTOLOGY_LEVEL_MAP.values())

    def score_ancestor_by_level(ancestor_level: OntologyLevel) -> int:
        diff = ONTOLOGY_LEVEL_MAP[ancestor_level] - ONTOLOGY_LEVEL_MAP[child_level]

        # if the ancestor is less specific than the child, disqualify it
        if diff < 0:
            return -1

        # smaller diffs are better
        return (MAX_ONTOLOGY_SCORE - diff) + 1

    return score_ancestor_by_level


def get_ancestor_type_scorer(
    child_types: Sequence[str],
) -> Callable[[list[str]], int]:
    """
    Returns an ancestor type scorer function

    Args:
        child_types (list[str]): list of child types

    Returns (Callable[[UmlsInfo], int]): a function that scores an ancestor by type
        higher is better; -1 means disqualified
    """
    available_ancestor_types: Mapping[str, int] = reduce(
        lambda a, b: {**a, **b},
        compact([PREFERRED_ANCESTOR_TYPE_MAP.get(ct) for ct in child_types]),
        {},
    )

    def score_ancestor_by_type(ancestor_types: list[str]) -> int:
        if not isinstance(ancestor_types, list):
            raise TypeError("ancestor_types must be a sequence")

        scores = compact([available_ancestor_types.get(at) for at in ancestor_types])
        if len(scores) == 0:
            return -1

        return max(scores)

    return score_ancestor_by_type


def increment_ontology_level(level: OntologyLevel) -> OntologyLevel:
    """
    Increment an ontology level
    """
    if level == OntologyLevel.SUBINSTANCE:
        return OntologyLevel.INSTANCE
    if level == OntologyLevel.INSTANCE:
        return OntologyLevel.L1_CATEGORY
    if level == OntologyLevel.L1_CATEGORY:
        return OntologyLevel.L2_CATEGORY
    if level == OntologyLevel.L2_CATEGORY:
        return OntologyLevel.L3_CATEGORY
    if level == OntologyLevel.L3_CATEGORY:
        return OntologyLevel.L4_CATEGORY
    if level == OntologyLevel.L4_CATEGORY:
        return OntologyLevel.L5_CATEGORY
    if level == OntologyLevel.L5_CATEGORY:
        return OntologyLevel.L5_CATEGORY

    logger.warning(f"Cannot increment level {level}, returning UNKNOWN")
    return OntologyLevel.UNKNOWN
