from dataclasses import dataclass
from typing import Optional, TypeGuard
import torch
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass(frozen=True)
class InputCategorySizes:
    multi_select: Optional[dict[str, int]] = None
    single_select: Optional[dict[str, int]] = None
    text: Optional[dict[str, int]] = None


@dataclass(frozen=True)
class OutputCategorySizes:
    y1: dict[str, int]
    y2: int


@dataclass(frozen=True)
class AllCategorySizes(InputCategorySizes, OutputCategorySizes):
    pass


CategorySizes = AllCategorySizes | InputCategorySizes


class TolerantDataclass:
    """
    A dataclass that can be instantiated with unknown kwargs (which are ignored)
    """

    @classmethod
    def _get_valid_args(cls, **kwargs):
        cls_attrs = cls.__dict__["__dataclass_fields__"].keys()
        _kwargs = {k: v for k, v in kwargs.items() if k in cls_attrs}
        if len(_kwargs) < len(kwargs):
            logger.debug(
                f"Instantiating {cls.__name__} with unknown kwargs: {kwargs.keys() - _kwargs.keys()}"
            )
        return _kwargs

    @classmethod
    def get_instance(cls, **kwargs):
        return cls(**cls._get_valid_args(**kwargs))

    # allow this to act like a dict
    def __getattr__(self, name):
        return getattr(self.__dict__, name)


@dataclass(frozen=True)
class ModelInput(TolerantDataclass):
    multi_select: torch.Tensor
    quantitative: torch.Tensor
    single_select: torch.Tensor
    text: torch.Tensor


@dataclass(frozen=True)
class SplitModelInput(TolerantDataclass):
    multi_select: list[torch.Tensor]
    quantitative: torch.Tensor
    single_select: list[torch.Tensor]
    text: torch.Tensor


@dataclass(frozen=True)
class ModelOutput(TolerantDataclass):
    y1_true: torch.Tensor  # embedded y1_true
    y2_true: torch.Tensor
    y2_oh_true: torch.Tensor


@dataclass(frozen=True)
class ModelInputAndOutput(ModelInput, ModelOutput):
    pass


@dataclass(frozen=True)
class SplitModelInputAndOutput(SplitModelInput, ModelOutput):
    pass


def is_model_input(
    model_params: ModelInput | ModelOutput | ModelInputAndOutput,
) -> TypeGuard[ModelInput]:
    if is_model_input_output(model_params):
        return False
    fields = model_params.keys()
    input_fields = ModelInput.__dataclass_fields__.keys()
    return set(input_fields).issubset(fields)


def is_model_output(
    model_params: ModelInput | ModelOutput | ModelInputAndOutput,
) -> TypeGuard[ModelOutput]:
    if is_model_input_output(model_params):
        return False
    fields = model_params.keys()
    output_fields = ModelOutput.__dataclass_fields__.keys()
    return set(output_fields).issubset(fields)


def is_model_input_output(
    model_params: ModelInput | ModelOutput | ModelInputAndOutput,
) -> TypeGuard[ModelInputAndOutput]:
    fields = model_params.keys()
    input_fields = ModelInput.__dataclass_fields__.keys()
    output_fields = ModelOutput.__dataclass_fields__.keys()
    return set(input_fields).issubset(fields) and set(output_fields).issubset(fields)


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
