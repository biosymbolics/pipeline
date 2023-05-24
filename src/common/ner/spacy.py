"""
Named-entity recognition using spacy
"""
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.tokens import Span
from spacy.language import Language
from pydash import compact, flatten

from common.ner.types import is_sci_spacy_linker
from constants.umls import UMLS_PHARMACOLOGIC_INTERVENTION_TYPES
from .types import KbLinker

ENTITY_TYPES = ["PRODUCT"]

SUMMARY_ATTRIBUTES = ["text", "label_", "kb_id_", "canonical_name", "types"]


def __summarize(entities: list[tuple]):
    for ent in entities:
        for attr in SUMMARY_ATTRIBUTES:
            if hasattr(ent, attr):
                print(f"Entity {attr} : {getattr(ent, attr)}")


def __get_kb_linker(nlp: Language) -> KbLinker:
    """
    Get the KB linker from the nlp pipeline
    """
    linker = nlp.get_pipe("scispacy_linker")

    if not is_sci_spacy_linker(linker):
        raise Exception("Invalid linker")

    return linker.kb


def __has_intersection(list_a, list_b):
    return any(elem in list_a for elem in list_b)


def __get_canonical_entities(entities: list[Span], kb_linker: KbLinker) -> dict:
    """
    Get canonical entities from the entities
    """
    canonical_entity_map = {}
    for entity in entities:
        kb_entities = [
            kb_linker.cui_to_entity[kb_ent[0]] for kb_ent in entity._.kb_ents
        ]
        canonical_entities = [
            ent.canonical_name
            for ent in kb_entities
            if __has_intersection(
                ent.types, UMLS_PHARMACOLOGIC_INTERVENTION_TYPES.keys()
            )
        ]
        if len(canonical_entities) > 0:
            canonical_entity_map[entity.text] = canonical_entities

    return canonical_entity_map


def extract_named_entities(content: list[str]) -> list[str]:
    """
    Extract named entities from a list of content

    Args:
        content (list[str]): list of content on which to do NER
    """
    nlp = spacy.load("en_core_sci_lg")  # en_core_web_trf
    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": 0.7,
            "filter_for_definitions": False,
            "no_definition_threshold": 0.9,
        },
    )
    entities = flatten([nlp(batch).ents for batch in content])

    linker = __get_kb_linker(nlp)
    canonical_entities = __get_canonical_entities(entities, linker)

    # __summarize(entities)
    # __summarize(kb_entities)
    print(canonical_entities)

    entity_strings = [
        entity.text for entity in entities if entity.label_ in ENTITY_TYPES
    ]
    return entity_strings
