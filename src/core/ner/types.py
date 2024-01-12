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
    Optional,
    TypeGuard,
    TypedDict,
    Union,
)
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy.tokens import Doc
from prisma.enums import BiomedicalEntityType

from constants.umls import UMLS_TO_ENTITY_TYPE
from typings.core import Dataclass


NerResult = TypedDict("NerResult", {"word": str, "score": float, "entity_group": str})


@dataclass(frozen=True)
class CanonicalEntity(Dataclass):
    id: str | None
    name: str
    ids: Optional[list[str]] = None
    description: Optional[str] = None
    aliases: list[str] = field(default_factory=list)
    types: list[str] = field(default_factory=list)

    @property
    def is_fake(self):
        # cheesy; we set id to name if it's a fake entity
        return self.name == self.id

    @property
    def type(self) -> BiomedicalEntityType:
        if len(self.types) == 0 or self.types[0] not in UMLS_TO_ENTITY_TYPE:
            return BiomedicalEntityType.UNKNOWN

        # if it is already a BiomedicalEntityType, return it
        if BiomedicalEntityType.__members__.get(self.types[0]):
            return BiomedicalEntityType[self.types[0]]

        # otherwise assume it is a TUI (UMLS) type and do a lookup
        return UMLS_TO_ENTITY_TYPE[self.types[0]]


@dataclass(frozen=True)
class DocEntity(Dataclass):
    term: str
    start_char: int
    end_char: int
    normalized_term: str
    type: str | None = None
    vector: Optional[list[float]] = None
    spacy_doc: Optional[Doc] = None
    canonical_entity: Optional[CanonicalEntity] = None

    def __post_init__(self):
        object.__setattr__(self, "type", self.get_type())
        object.__setattr__(self, "vector", self.get_vector())

    @property
    def id(self) -> Optional[str]:
        if self.canonical_entity is None:
            return None
        return self.canonical_entity.id

    @property
    def canonical_name(self):
        if self.canonical_entity is None:
            return None
        return self.canonical_entity.name

    def __str__(self):
        return f"{self.term} ({self.type}, s: {self.start_char}, e: {self.end_char}, norm_term: {self.canonical_name or self.normalized_term}, id: {self.id})"

    def __repr__(self):
        return self.__str__()

    def get_vector(self):
        if self.vector is not None:
            return self.vector
        if self.spacy_doc is not None:
            return self.spacy_doc.vector.tolist()
        return None

    def get_type(self):
        if self.type is not None:
            return self.type
        if self.canonical_entity is not None:
            return self.canonical_entity.type
        return None

    def to_flat_dict(self):
        """
        Convert to a flat dictionary
        (Skips canonical_entity)
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
