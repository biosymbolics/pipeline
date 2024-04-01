"""
NER module
"""

# from .classifier import classify_by_keywords
from .ner import NerTagger
from .normalizer import TermNormalizer

__all__ = [
    "NerTagger",
    "TermNormalizer",
]
