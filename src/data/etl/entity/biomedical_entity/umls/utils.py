from functools import reduce
from typing import Callable, Mapping, Sequence
from pydash import compact, flatten, uniq
from prisma.enums import OntologyLevel
import logging

from constants.umls import PREFERRED_ANCESTOR_TYPE_MAP

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
    score_ancestor_by_type = get_ancestor_type_scorer(child.type_ids)

    best_ancestors = sorted(
        [a for a in ancestors if score_ancestor_by_type(a.type_ids) != -1],
        key=lambda a: score_ancestor_by_type(a.type_ids),
        reverse=True,
    )

    return best_ancestors[0]


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


def choose_best_ancestor_type(
    child_types: Sequence[str], ancestor_types: Sequence[str]
) -> str | None:
    """
    Choose the best available ancestor type for a child from a list of possible ancestor types

    Args:
        child_types (list[str]): list of child types
        ancestor_types (list[str]): list of possible ancestor types

    Returns (str): best ancestor type (a tui)
    """
    score_ancestor_by_type = get_ancestor_type_scorer(child_types)
    types_by_preference = flatten(
        sorted(
            [[at] for at in ancestor_types], key=score_ancestor_by_type, reverse=True
        )
    )

    if len(types_by_preference) == 0:
        return None

    return types_by_preference[0]  # type: ignore


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
