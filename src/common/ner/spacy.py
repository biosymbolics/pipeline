"""
Named-entity recognition using spacy
"""
import spacy
from spacy.tokens import Span
from spacy.language import Language
from pydash import flatten

from common.ner.types import is_sci_spacy_linker
from .types import KbLinker

ENTITY_TYPES = ["PRODUCT"]

SUMMARY_ATTRIBUTES = ["text", "label_", "kb_id_"]


def __summarize(entities: list[Span]):
    for ent in entities:
        for attr in SUMMARY_ATTRIBUTES:
            try:
                if hasattr(ent, attr):
                    print(f"Entity {attr} : {getattr(ent, attr)}")
            except Exception as ex:
                print("tried %s", ex)


def __get_kb_linker(nlp: Language) -> KbLinker:
    """
    Get the KB linker from the nlp pipeline
    """
    linker = nlp.get_pipe("scispacy_linker")

    if not is_sci_spacy_linker(linker):
        raise Exception("Invalid linker")

    kb_linker = linker.get("kb")
    return kb_linker


def get_kb_entities(entities: list[Span], kb_linker: KbLinker) -> list[Span]:
    """
    Get the KB entities from the entities
    """
    kb_ents = flatten([entity._.kb_ents for entity in entities])
    linked_ents = [kb_linker.cui_to_entity[umls_ent[0]] for umls_ent in kb_ents]  # type: ignore
    return linked_ents


def extract_named_entities(content: list[str]) -> list[str]:
    """
    Extract named entities from a list of content

    Args:
        content (list[str]): list of content on which to do NER
    """
    nlp = spacy.load("en_core_sci_sm")  # en_core_web_trf
    nlp.add_pipe(
        "scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"}
    )
    entities = flatten([nlp(batch).ents for batch in content])

    linker = __get_kb_linker(nlp)
    kb_entities = get_kb_entities(entities, linker)

    __summarize(entities)
    __summarize(kb_entities)

    entity_strings = [
        entity.text if entity.label_ in ENTITY_TYPES else "" for entity in entities
    ]
    return entity_strings
