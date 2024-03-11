"""
NER types
"""

from typing import (
    Any,
    Collection,
    Mapping,
    Optional,
    TypeGuard,
    TypedDict,
    Union,
)
from pydantic import BaseModel, ConfigDict, SkipValidation
from pydash import compact
from spacy.tokens import Doc
from prisma.enums import BiomedicalEntityType

from data.domain.biomedical.umls import tuis_to_entity_type


class CanonicalEntity(BaseModel):
    id: str | None
    name: str
    ids: list[str] | None = None
    description: str | None = None
    aliases: list[str] = []
    types: list[str] = []

    @property
    def is_fake(self):
        # cheesy; we set id to name if it's a fake entity
        return self.name == self.id

    @property
    def type(self) -> BiomedicalEntityType:
        # TODO: should take into consideration type in DocEntity, when thusly instantiated.
        # (UMLS is an ok proxy for type, but it gets messy, e.g. bacteria - intervention or disease? context tells us.)
        # if any already BiomedicalEntityTypes, return one
        bets = compact([BiomedicalEntityType.__members__.get(t) for t in self.types])

        if len(bets) > 0:
            return bets[0]

        return tuis_to_entity_type(self.types)


class DocEntity(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    term: str
    start_char: int
    end_char: int
    normalized_term: str
    type: str | None = None
    vector: SkipValidation[list[float] | None] = None
    spacy_doc: SkipValidation[Doc | None] = None
    canonical_entity: SkipValidation[CanonicalEntity | None] = None

    @staticmethod
    def create(
        term: str,
        normalized_term=None,
        start_char=0,
        end_char=0,
        type=None,
        vector: list[float] | None = None,
        spacy_doc=None,
        canonical_entity=None,
    ):
        def _vector() -> list[float] | None:
            if vector is not None and isinstance(vector, list):
                return vector
            elif spacy_doc is not None:
                return spacy_doc.vector
            return None

        return DocEntity(
            term=term,
            start_char=start_char,
            end_char=end_char,
            normalized_term=normalized_term or term,
            type=type
            or None,  # (getattr(spacy_doc, "label_", None) if spacy_doc else None),
            vector=_vector(),
            spacy_doc=spacy_doc,
            canonical_entity=canonical_entity,
        )

    @staticmethod
    def merge(
        e: "DocEntity",
        **kwargs,
    ):
        return DocEntity(
            **{k: v for k, v in e.model_dump().items() if k not in kwargs},
            **kwargs,
        )

    def copy(
        self,
        **kwargs,
    ):
        return DocEntity(
            term=kwargs.get("term", self.term),
            start_char=kwargs.get("start_char", self.start_char),
            end_char=kwargs.get("end_char", self.end_char),
            normalized_term=kwargs.get("normalized_term", self.normalized_term),
            type=kwargs.get("type", self.type),
            vector=kwargs.get("vector", self.vector),
            spacy_doc=kwargs.get("spacy_doc", self.spacy_doc),
            canonical_entity=kwargs.get("canonical_entity", self.canonical_entity),
        )

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
