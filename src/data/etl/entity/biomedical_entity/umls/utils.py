from functools import reduce
from typing import Mapping, Sequence
from pydash import compact
from prisma.enums import OntologyLevel

from constants.umls import PREFERRED_ANCESTOR_TYPE_MAP

from .types import UmlsInfo, compare_ontology_level


def choose_max_ontology_level(
    a: OntologyLevel | None, b: OntologyLevel | None
) -> OntologyLevel | None:
    if a is None:
        return b
    if b is None:
        return a

    return a if compare_ontology_level(a, b) >= 0 else b


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
        return OntologyLevel.L2_CATEGORY

    raise ValueError(f"Cannot increment level {level}")


def choose_best_available_ancestor(
    child_types: list[str], ancestors: Sequence[UmlsInfo]
) -> UmlsInfo | None:
    """
    Choose the best available ancestor

    Args:
        child_types (list[str]): list of child types
        ancestors (list[str]): list of ancestor
    """
    # create a map of all possible parent types
    possible_ancestor_types = {
        type_id: a
        # sorted so that closer ancestors are preferred
        for type_id, a in sorted(
            [(type_id, a) for a in ancestors for type_id in a.type_ids],
            reverse=True,
        )
    }

    best_type = choose_best_available_ancestor_type(
        child_types, list(possible_ancestor_types.keys())
    )

    if best_type is None:
        return None

    return possible_ancestor_types[best_type]


def choose_best_available_ancestor_type(
    child_types: Sequence[str], ancestor_types: Sequence[str]
) -> str | None:
    """
    Choose the best available ancestor type for a child from a list of possible ancestor types

    Args:
        child_types (list[str]): list of child types
        ancestor_types (list[str]): list of possible parent types
    """
    # create a map of all preferred types for child
    preferred_type_map: Mapping[str, int] = reduce(
        lambda a, b: {**a, **b},
        compact([PREFERRED_ANCESTOR_TYPE_MAP.get(ct) for ct in child_types]),
    )
    # sort possible parent types by preference
    types_by_preference = sorted(
        zip(
            ancestor_types,
            [preferred_type_map.get(t, 1000) for t in ancestor_types],
        ),
        key=lambda x: x[1],
    )

    if len(types_by_preference) == 0:
        return None

    preferred_type = types_by_preference[0][0]
    return preferred_type
