from dataclasses import dataclass
from typing import TypeGuard, TypedDict


ModelMetrics = TypedDict(
    "ModelMetrics", {"precision": float, "recall": float, "f1": float}
)


@dataclass(frozen=True)
class InputFieldLists:
    single_select: list[str]
    multi_select: list[str]
    text: list[str]
    quantitative: list[str]


@dataclass(frozen=True)
class FieldLists(InputFieldLists):
    y1_categorical: list[str]  # e.g. cat fields to predict, e.g. randomization
    y2: str


def is_all_fields_list(
    list: FieldLists | InputFieldLists,
) -> TypeGuard[FieldLists]:
    fields = list.__dict__.keys()
    res = "y1_categorical" in fields and "y2" in fields
    return res
