from typing import Sequence
from pydash import flatten, uniq_by

from constants.patterns.iupac import is_iupac
from constants.umls import MOST_PREFERRED_UMLS_TYPES
from core.ner.types import CanonicalEntity
from data.domain.biomedical.umls import clean_umls_name
from utils.list import has_intersection


MIN_WORD_LENGTH = 1


def is_composite_eligible(term: str, min_word_length: int = MIN_WORD_LENGTH) -> bool:
    """
    Is a text a composite candidate?

    - False if it's an IUPAC name, which are just too easy to mangle (e.g. 3'-azido-2',3'-dideoxyuridine matching 'C little e')
    - false if it's too short (a single token or word)
    - Otherwise true
    """
    if is_iupac(term):
        return False
    if len(term.split(" ")) <= min_word_length:
        return False
    return True


def form_composite_name(members: Sequence[CanonicalEntity]) -> str:
    """
    Form a composite name from the entities from which it is comprised
    """

    def get_name_part(ce: CanonicalEntity):
        return clean_umls_name(ce.id or ce.name, ce.name, ce.aliases, ce.types, True)

    name = " ".join([get_name_part(c) for c in members])
    return name


def form_composite_entity(members: Sequence[CanonicalEntity]) -> CanonicalEntity:
    """
    Form a composite from a list of member entities
    """

    # if just a single composite match, treat it like a non-composite match
    if len(members) == 1:
        return members[0]

    selected_members = select_composite_members(members)

    # sorted for the sake of consist composite ids
    ids = sorted([m.id for m in selected_members if m.id is not None])

    # types (CanonicalEntity class will infer single type from UMLS TUIs)
    types = flatten([m.types for m in selected_members])

    # form name from comprising candidates
    name = form_composite_name(selected_members)

    return CanonicalEntity(
        id="|".join(ids),
        ids=ids,
        name=name,
        types=types,
        # description=..., # TODO: composite description
        # aliases=... # TODO: all permutations
    )


def select_composite_members(
    _members: Sequence[CanonicalEntity],
) -> list[CanonicalEntity]:
    """
    Select composite members to return
    """
    members = uniq_by(_members, lambda m: m.id)
    real_members = [m for m in members if not m.is_fake]

    if len(real_members) == 0:
        return list(members)

    # Partial match if non-matched words, and only a single candidate (TODO: revisit)
    is_partial = (
        # has 1+ fake members (i.e. unmatched)
        len(real_members) < len(members)
        # and only one real candidate match
        and len(real_members) == 1
    )

    # if partial match, include *all* candidates, which includes the faked ones
    # "UNMATCHED inhibitor" will have a name and id that reflects the unmatched word
    if is_partial:
        return list(members)

    # if we have 1+ preferred candidates, return those
    # this prevents composites like C0024579|C0441833 ("Maleimides Groups") - wherein "Group" offers little value
    preferred = [
        m
        for m in real_members
        if has_intersection(m.types, list(MOST_PREFERRED_UMLS_TYPES.keys()))
    ]
    if len(preferred) >= 1:
        return preferred

    # else, we're going to drop unmatched words
    # e.g. "cpla (2)-selective inhibitor" -> "cpla inhibitor"

    return real_members
