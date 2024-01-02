"""
NER types
"""
from dataclasses import dataclass, field
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
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from constants.umls import UMLS_TO_ENTITY_TYPE


from typings.core import Dataclass


NerResult = TypedDict("NerResult", {"word": str, "score": float, "entity_group": str})


@dataclass(frozen=True)
class CanonicalEntity(Dataclass):
    id: str
    name: str
    ids: Optional[list[str]] = None
    description: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)

    def type(self) -> Optional[str]:
        if len(self.types) == 0 or self.types[0] not in UMLS_TO_ENTITY_TYPE:
            return None
        return UMLS_TO_ENTITY_TYPE[self.types[0]]


@dataclass(frozen=True)
class DocEntity(Dataclass):
    term: str
    type: str
    start_char: int
    end_char: int
    normalized_term: str
    # embeddings of mention
    vector: Optional[List[float]] = None
    spacy_doc: Optional[Doc] = None
    linked_entity: Optional[CanonicalEntity] = None

    @property
    def id(self) -> Optional[str]:
        if self.linked_entity is None:
            return None
        return self.linked_entity.id

    @property
    def canonical_name(self):
        if self.linked_entity is None:
            return None
        return self.linked_entity.name

    def __str__(self):
        return f"{self.term} ({self.type}, s: {self.start_char}, e: {self.end_char}, norm_term: {self.canonical_name or self.normalized_term}, id: {self.id})"

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
            "vector": self.vector,
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
