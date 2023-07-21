from typing import Optional, TypedDict


class Annotation(TypedDict):
    """
    Annotation class
    """

    id: str
    entity_type: str
    start_char: int
    end_char: int
    text: str


Feature = TypedDict(
    "Feature",
    {
        "id": str,
        "text": str,
        "offset_mapping": list[Optional[tuple[int, int]]],
        "token_start_mask": list[int],
        "token_end_mask": list[int],
    },
)
