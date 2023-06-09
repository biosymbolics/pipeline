"""
Model-based NER
"""
from common.utils.list import dedup

from . import spacy as spacy_ner


def extract_named_entities(content: list[str]) -> list[str]:
    """
    Extract named entities from a list of content

    Args:
        content (list[str]): list of content on which to do NER
    """
    all_entities = spacy_ner.extract_named_entities(content)
    return dedup(all_entities)
