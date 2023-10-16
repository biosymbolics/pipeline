from typing import NamedTuple, TypedDict


ModelMetrics = TypedDict(
    "ModelMetrics", {"precision": float, "recall": float, "f1": float}
)


# keep in sync with FieldLists
class InputFieldLists(NamedTuple):
    single_select: list[str]
    multi_select: list[str]
    text: list[str]
    quantitative: list[str]


# keep in sync with InputFieldLists
# sadly, FieldLists(InputFieldLists) doesn't work.
class FieldLists(NamedTuple):
    single_select: list[str]
    multi_select: list[str]
    text: list[str]
    quantitative: list[str]
    y1_categorical: list[str]  # e.g. cat fields to predict, e.g. randomization
    y2: str
