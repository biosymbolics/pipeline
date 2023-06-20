"""
NER module
"""
from .classifier import classify_by_keywords
from .ner import NerTagger
from .normalizer import TermNormalizer, NormalizationMap

__all__ = [
    "classify_by_keywords",
    "NerTagger",
    "NormalizationMap",
    "TermNormalizer",
]
