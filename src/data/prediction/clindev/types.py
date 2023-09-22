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
    single_select_input: int
    text_input: int
    stage1_input: int
    stage2_input: int
    stage1_hidden: int  # = 64
    stage1_embedded_output: int  # = 32
    stage1_output_map: dict[str, int]
    stage2_hidden: int  # = 64
    stage2_output: int  # = 1
    quantitative_input: int
