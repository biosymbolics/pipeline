from typing import TypedDict
import torch


DnnInput = TypedDict(
    "DnnInput",
    {
        "x1": torch.Tensor,
        "x1_inputs": torch.Tensor,
        "y1": torch.Tensor,
        "y2": torch.Tensor,
    },
)
