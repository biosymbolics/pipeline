from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class InputCategorySizes:
    multi_select: dict[str, int]
    single_select: dict[str, int]


@dataclass(frozen=True)
class AllCategorySizes(InputCategorySizes):
    y1: dict[str, int]
    y2: int


CategorySizes = AllCategorySizes | InputCategorySizes


@dataclass(frozen=True)
class ModelInput:
    multi_select: torch.Tensor
    quantitative: torch.Tensor
    single_select: torch.Tensor
    text: torch.Tensor


@dataclass(frozen=True)
class ModelInputAndOutput(ModelInput):
    #  y1_categories: torch.Tensor  # used as y1_true (encodings)
    y1_true: torch.Tensor  # embedded y1_true
    y2_true: torch.Tensor
    y2_oh_true: torch.Tensor
