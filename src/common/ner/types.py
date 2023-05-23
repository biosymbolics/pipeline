from typing import Any, TypeGuard, TypedDict


NerResult = TypedDict("NerResult", {"word": str, "score": float, "entity_group": str})


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
