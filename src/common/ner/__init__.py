"""
NER module
"""
# from .classifier import classify_by_keywords
# from .ner import NerTagger
from .linker import TermNormalizer, TermLinker

__all__ = [
    "TermLinker",
    "TermNormalizer",
]
