"""
NER module
"""
# from .classifier import classify_by_keywords
# from .ner import NerTagger
from .linker import TermLinker, LinkedEntityMap

__all__ = [
    "TermLinker",
    "LinkedEntityMap",
]
