import math
from typing import NamedTuple, TypedDict
import torch


class AllCategorySizes(NamedTuple):
    multi_select: dict[str, int]
    single_select: dict[str, int]
    y1: dict[str, int]


DnnInput = TypedDict(
    "DnnInput",
    {
        "multi_select_x": torch.Tensor,
        "single_select_x": torch.Tensor,
        "text_x": torch.Tensor,  # can be empty
        "quantitative_x": torch.Tensor,  # can be empty
        "y1_categories": torch.Tensor,  # used as y1_true (encodings)
        "y1": torch.Tensor,  # embedded y1_true
        "y2": torch.Tensor,
    },
)


class TwoStageModelSizes(NamedTuple):
    """
    Sizes of inputs and outputs for two-stage model
    """

    categories_by_field: AllCategorySizes
    multi_select_input: int
    quantitative_input: int
    single_select_input: int
    text_input: int
    stage1_output: int
    stage1_output_map: dict[str, int]
    stage2_output: int
    embedding_dim: int

    @property
    def multi_select_output(self):
        return round(
            (self.multi_select_input / 5)  # shrink multis, many of which are zero
            * self.embedding_dim
            * math.log2(sum(self.categories_by_field.multi_select.values()))
        )

    @property
    def single_select_output(self):
        return round(
            self.single_select_input
            * self.embedding_dim
            * math.log2(sum(self.categories_by_field.single_select.values()))
        )

    @property
    def stage1_input(self):
        return (
            self.multi_select_output
            + self.quantitative_input
            + self.single_select_output
            + self.text_input
        )

    @property
    def stage1_hidden(self):
        return round(self.stage1_input * 3)

    @property
    def stage2_input(self):
        return self.stage1_input

    @property
    def stage2_hidden(self):
        return round(self.stage2_input * 0.5)

    def __str__(self):
        return (
            " ".join([f"{f}: {v}" for f, v in self._asdict().items()])
            + " "
            + f"multi_select_output: {self.multi_select_output} "
            f"single_select_output: {self.single_select_output} "
            f"stage1_input: {self.stage1_input} "
            f"stage1_hidden: {self.stage1_hidden} "
            f"stage2_input: {self.stage2_input} "
            f"stage2_hidden: {self.stage2_hidden} "
        )
