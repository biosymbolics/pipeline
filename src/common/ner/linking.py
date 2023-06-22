"""
Utils for linking canonical entities
"""
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.tokens import Span
from spacy.language import Language

from constants.umls import UMLS_PHARMACOLOGIC_INTERVENTION_TYPES
from common.ner.types import is_sci_spacy_linker
from common.utils.list import has_intersection

from .types import KbLinker

ENTITY_TYPES = ["PRODUCT", "COMPOUND", "MECHANISM", "DISEASE"]


def __get_kb_linker(nlp: Language) -> KbLinker:
    """
    Get the KB linker from the nlp pipeline
    """
    linker = nlp.get_pipe("scispacy_linker")

    if not is_sci_spacy_linker(linker):
        raise ValueError("linker is not a SciSpacyLinker")

    return linker.kb


def enrich_with_canonical(entities: list[Span], nlp: Language) -> dict[str, list[str]]:
    """
    Links canonical entities if possible

    Args:
        entities (list[Span]): list of entities
        kb_linker (KbLinker): KB linker

    Currently only for PRODUCT entities
    """
    linker = __get_kb_linker(nlp)

    _entities = [entity for entity in entities if entity.label_ in ENTITY_TYPES]

    canonical_entity_map = {}
    for entity in _entities:
        kb_entities = [linker.cui_to_entity[kb_ent[0]] for kb_ent in entity._.kb_ents]
        canonical_entities = [
            ent.canonical_name
            for ent in kb_entities
            if has_intersection(
                ent.types, list(UMLS_PHARMACOLOGIC_INTERVENTION_TYPES.keys())
            )
        ]
        canonical_entity_map[entity.text] = canonical_entities

    return canonical_entity_map
