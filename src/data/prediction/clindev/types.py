from typing import NamedTuple, TypedDict
import torch


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

    multi_select_input: int
    quantitative_input: int
    single_select_input: int
    text_input: int
    stage1_output: int
    stage1_output_map: dict[str, int]
    stage2_output: int

    @property
    def stage1_input(self):
        return (
            self.multi_select_input
            + self.quantitative_input
            + self.single_select_input
            + self.text_input
        )

    @property
    def stage1_hidden(self):
        return round(self.stage1_input * (2 / 3))

    @property
    def stage2_hidden(self):
        return round(self.stage1_input * (2 / 3))

    @property
    def stage2_input(self):
        return self.stage1_input
