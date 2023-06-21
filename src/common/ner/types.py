"""
NER types
"""
from typing import (
    Any,
    Callable,
    Collection,
    Mapping,
    NamedTuple,
    Optional,
    TypeGuard,
    TypedDict,
    Union,
)
from spacy.language import Language
from spacy.pipeline import Pipe
from spacy.tokens import Token
from spacy.tokenizer import Tokenizer


NerResult = TypedDict("NerResult", {"word": str, "score": float, "entity_group": str})


class SciSpacyEntity(NamedTuple):
    concept_id: str
    canonical_name: str
    aliases: list[str]
    types: list[str]
    definition: Optional[str]


class KbLinker(NamedTuple):
    cui_to_entity: dict[Token, SciSpacyEntity]


class SciSpacyLinker(NamedTuple):
    kb: KbLinker


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
