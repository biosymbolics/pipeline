"""
NER types
"""
from collections import namedtuple
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Collection,
    List,
    Mapping,
    NamedTuple,
    Optional,
    TypeGuard,
    TypedDict,
    Union,
)
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokenizer import Tokenizer

from typings.core import Dataclass


NerResult = TypedDict("NerResult", {"word": str, "score": float, "entity_group": str})


class CanonicalEntity(NamedTuple):
    id: str
    name: str
    ids: Optional[list[str]] = None
    description: Optional[str] = None
    aliases: list[str] = []
    types: list[str] = []


@dataclass(frozen=True)
class DocEntity(Dataclass):
    term: str
    type: str
    start_char: int
    end_char: int
    normalized_term: str
    linked_entity: Optional[CanonicalEntity]
    span_embeddings: Optional[List[float]] = None

    def __str__(self):
        norm_term = (
            self.linked_entity.name if self.linked_entity else self.normalized_term
        )
        return f"{self.term} ({self.type}, s: {self.start_char}, e: {self.end_char}, norm_term: {norm_term})"

    def __repr__(self):
        return self.__str__()

    def to_flat_dict(self):
        """
        Convert to a flat dictionary
        (Skips linked_entity)
        """
        return {
            "term": self.term,
            "type": self.type,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "normalized_term": self.normalized_term,
        }


DocEntities = list[DocEntity]


def is_entity_doc_list(obj: Any) -> TypeGuard[DocEntities]:
    """
    Check if object is a list of entities
    """
    return isinstance(obj, list) and len(obj) > 0 and isinstance(obj[0], DocEntity)


SpacyPattern = Mapping[str, Union[str, Collection[Mapping[str, Any]]]]
SpacyPatterns = Collection[SpacyPattern]


def is_ner_result(entity: Any) -> TypeGuard[NerResult]:
    """
    Check if entity is a valid NER result
    """
    return (
        isinstance(entity, dict)
        and entity.get("word") is not None
        and entity.get("score") is not None
        and entity.get("entity_group") is not None
    )


GetTokenizer = Callable[[Language], Tokenizer]

SynonymRecord = TypedDict("SynonymRecord", {"term": str, "synonym": str})
