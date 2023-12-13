from dataclasses import dataclass
from typing import Optional, TypedDict
import torch

from typings.core import Dataclass


@dataclass(frozen=True)
class Annotation(Dataclass):
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
        "token_start_mask": torch.Tensor,
        "token_end_mask": torch.Tensor,
    },
)
