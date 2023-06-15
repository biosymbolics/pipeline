"""
NER module
"""
from .classifier import classify_by_keywords
from .ner import extract_named_entities
from .normalizer import TermNormalizer, NormalizationMap

__all__ = [
    "classify_by_keywords",
    "extract_named_entities",
    "NormalizationMap",
    "TermNormalizer",
]
