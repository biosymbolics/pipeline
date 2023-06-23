"""
NER module
"""
from .classifier import classify_by_keywords
from .ner import NerTagger, tagger
from .normalizer import TermNormalizer, NormalizationMap

__all__ = [
    "classify_by_keywords",
    "tagger",
    "NerTagger",
    "NormalizationMap",
    "TermNormalizer",
]
