from typing import Optional, NamedTuple, TypedDict
import torch


DnnInput = TypedDict(
    "DnnInput",
    {
        "multi_select_x": torch.Tensor,
        "single_select_x": torch.Tensor,
        "text_x": Optional[torch.Tensor],
        "y1": torch.Tensor,
        "y1_categories": torch.Tensor,
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
    stage1_prob_output: int  # = 32
    stage2_hidden: int  # = 64
    stage2_output: int  # = 1
