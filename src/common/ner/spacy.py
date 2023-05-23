"""
Named-entity recognition using spacy
"""
import spacy
from spacy.tokens import Span
from pydash import flatten

from common.ner.types import is_sci_spacy_linker

ENTITY_TYPES = ["PRODUCT"]

ATTRIBUTES = ["text", "label_", "kb_id_"]


def __summarize(entities: list[Span]):
    for ent in entities:
        for attr in ATTRIBUTES:
            try:
                if hasattr(ent, attr):
                    print(f"Entity {attr} : {getattr(ent, attr)}")
            except Exception as ex:
                print("tried %s", ex)


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

    linker = nlp.get_pipe("scispacy_linker")

    if not is_sci_spacy_linker(linker):
        raise Exception("Invalid linker")

    kb_linker = linker.get("kb")
    kb_ents = flatten([entity._.kb_ents for entity in entities])
    for umls_ent in kb_ents:
        print(kb_linker.cui_to_entity[umls_ent[0]])  # type: ignore

    __summarize(entities)
    entity_strings = [
        entity.text if entity.label_ in ENTITY_TYPES else "" for entity in entities
    ]
    return entity_strings
