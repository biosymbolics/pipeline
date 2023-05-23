from typing import Any, Callable, TypeGuard, TypedDict

from spacy.pipeline import Pipe
from spacy.tokens import Span


NerResult = TypedDict("NerResult", {"word": str, "score": float, "entity_group": str})
KbLinker = TypedDict("KbLinker", {"cui_to_entity": Callable[[Span], list[Span]]})
SciSpacyLinker = TypedDict("SciSpacyLinker", {"kb": KbLinker})


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
    return isinstance(linker, dict) and linker.get("kb") is not None
