from typing import Sequence
from spacy.kb import KnowledgeBase

from constants.patterns.iupac import is_iupac
from core.ner.types import CanonicalEntity, DocEntity
from data.domain.biomedical.umls import clean_umls_name


MIN_WORD_LENGTH = 1


def is_composite_eligible(
    entity: DocEntity, min_word_length: int = MIN_WORD_LENGTH
) -> bool:
    """
    Is a text a composite candidate?

    - False if it's an IUPAC name, which are just too easy to mangle (e.g. 3'-azido-2',3'-dideoxyuridine matching 'C little e')
    - false if it's too short (a single token or word)
    - Otherwise true
    """
    tokens = entity.spacy_doc or entity.normalized_term.split(" ")
    if is_iupac(entity.normalized_term):
        return False
    if len(tokens) <= min_word_length:
        return False
    return True


def form_composite_name(members: Sequence[CanonicalEntity], kb: KnowledgeBase) -> str:
    """
    Form a composite name from the entities from which it is comprised
    """

    def get_name_part(c: CanonicalEntity):
        if c.id in kb.cui_to_entity:
            ce = kb.cui_to_entity[c.id]
            return clean_umls_name(
                ce.concept_id, ce.canonical_name, ce.aliases, ce.types, True
            )
        return c.name

    name = " ".join([get_name_part(c) for c in members])
    return name


def form_composite_entity(
    members: Sequence[CanonicalEntity], kb: KnowledgeBase
) -> CanonicalEntity:
    """
    Form a composite from a list of member entities
    """
    # if just a single composite match, treat it like a non-composite match
    if len(members) == 1:
        return members[0]

    # sorted for the sake of consist composite ids
    ids = sorted([m.id for m in members if m.id is not None])

    # form name from comprising candidates
    name = form_composite_name(members, kb)

    return CanonicalEntity(
        id="|".join(ids),
        ids=ids,
        name=name,
        # description=..., # TODO: composite description
        # aliases=... # TODO: all permutations
    )
