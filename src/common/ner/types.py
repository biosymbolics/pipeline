"""
NER types
"""
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


NerResult = TypedDict("NerResult", {"word": str, "score": float, "entity_group": str})


class CanonicalEntity(NamedTuple):
    id: str
    name: str
    aliases: Optional[List[str]] = []


DocEntity = tuple[str, str, int, int, Optional[CanonicalEntity]]
DocEntities = list[DocEntity]


class SpacyCanonicalEntity(NamedTuple):
    concept_id: str
    canonical_name: str
    aliases: List[str]
    types: List[str] = []
    definition: Optional[str] = None


class KbLinker(NamedTuple):
    cui_to_entity: dict[str, SpacyCanonicalEntity]


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


def is_sci_spacy_linker(linker: Pipe) -> TypeGuard[SciSpacyLinker]:
    """
    Check if entity is a valid SciSpacyLinker
    """
    return hasattr(linker, "kb") is not None


GetTokenizer = Callable[[Language], Tokenizer]
