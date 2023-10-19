from dataclasses import dataclass
from typing import TypeGuard
import torch


@dataclass(frozen=True)
class InputCategorySizes:
    multi_select: dict[str, int]
    single_select: dict[str, int]


@dataclass(frozen=True)
class OutputCategorySizes:
    y1: dict[str, int]
    y2: int


@dataclass(frozen=True)
class AllCategorySizes(InputCategorySizes, OutputCategorySizes):
    pass


CategorySizes = AllCategorySizes | InputCategorySizes


@dataclass(frozen=True)
class ModelInput:
    multi_select: torch.Tensor
    quantitative: torch.Tensor
    single_select: torch.Tensor
    text: torch.Tensor


@dataclass(frozen=True)
class ModelOutput:
    y1_true: torch.Tensor  # embedded y1_true
    y2_true: torch.Tensor
    y2_oh_true: torch.Tensor


@dataclass(frozen=True)
class ModelInputAndOutput(ModelInput, ModelOutput):
    pass


def is_model_output(
    model_params: ModelInput | ModelOutput | ModelInputAndOutput,
) -> TypeGuard[ModelOutput]:
    fields = model_params.__dict__.keys()
    return "y1_true" in fields and "y2_true" in fields and "y2_oh_true" in fields


@dataclass(frozen=True)
class InputFieldLists:
    single_select: list[str]
    multi_select: list[str]
    text: list[str]
    quantitative: list[str]


@dataclass(frozen=True)
class OutputFieldLists:
    y1_categorical: list[str]  # e.g. cat fields to predict, e.g. randomization
    y2: str


@dataclass(frozen=True)
class FieldLists(InputFieldLists, OutputFieldLists):
    pass


AnyFieldLists = FieldLists | InputFieldLists | OutputFieldLists


def is_all_fields_list(
    list: FieldLists | InputFieldLists,
) -> TypeGuard[FieldLists]:
    fields = list.__dict__.keys()
    return "y1_categorical" in fields and "y2" in fields
