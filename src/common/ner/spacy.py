"""
Named-entity recognition using spacy
"""
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.tokens import Span
from spacy.language import Language
from pydash import flatten

from common.ner.types import is_sci_spacy_linker
from constants.umls import UMLS_PHARMACOLOGIC_INTERVENTION_TYPES

from . import patterns
from .types import KbLinker

ENTITY_TYPES = ["PRODUCT"]


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
        canonical_entity_map[entity.text] = canonical_entities

    return canonical_entity_map


def extract_named_entities(content: list[str]) -> list[str]:
    """
    Extract named entities from a list of content

    Args:
        content (list[str]): list of content on which to do NER
    """
    # train_ner(content)
    nlp: Language = spacy.load("en_core_sci_lg")  # en_core_web_trf, en_core_sci_scibert
    ruler = nlp.add_pipe("entity_ruler", config={"validate": True}, before="ner")
    ruler.add_patterns([{"label": "PRODUCT", "pattern": patterns.MOA_PATTERN}])  # type: ignore

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
    analysis = nlp.analyze_pipes(pretty=True)
    print(analysis)
    entities = flatten([nlp(batch).ents for batch in content])
    chunks = flatten([nlp(batch).noun_chunks for batch in content])
    linker = __get_kb_linker(nlp)
    canonical_entities = __get_canonical_entities(entities, linker)

    for chunk in chunks:
        print(chunk)
    # __summarize(kb_entities)
    # print(canonical_entities)

    entity_strings = [
        entity.text for entity in entities if entity.label_ in ENTITY_TYPES
    ]
    return entity_strings
