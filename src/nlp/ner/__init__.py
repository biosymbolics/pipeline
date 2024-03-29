"""
NER module
"""
# from .classifier import classify_by_keywords
from .ner import NerTagger
from .linker.linker import TermLinker
from .normalizer import TermNormalizer

__all__ = [
    "NerTagger",
    "TermLinker",
    "TermNormalizer",
]
