"""
NER module
"""

# from .classifier import classify_by_keywords
from .ner import Ner
from .normalizer import TermNormalizer

__all__ = [
    "Ner",
    "TermNormalizer",
]
