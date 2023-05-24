"""
NER types
"""
from typing import Any, Callable, Mapping, NamedTuple, Optional, TypeGuard, TypedDict

from spacy.pipeline import Pipe
from spacy.tokens import Span, Token


NerResult = TypedDict("NerResult", {"word": str, "score": float, "entity_group": str})


class SciSpacyEntity(NamedTuple):
    concept_id: str
    canonical_name: str
    aliases: list[str]
    types: list[str]
    definition: Optional[str]


class KbLinker(NamedTuple):
    cui_to_entity: Mapping[Token, SciSpacyEntity]


class SciSpacyLinker(NamedTuple):
    kb: KbLinker


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


def is_sci_spacy_linker(linker: Pipe) -> TypeGuard[SciSpacyLinker]:
    """
    Check if entity is a valid SciSpacyLinker
    """
    return hasattr(linker, "kb") is not None
