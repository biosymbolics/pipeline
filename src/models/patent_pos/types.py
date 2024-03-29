from typing import TypedDict
import torch


GnnInput = TypedDict("GnnInput", {"x2": torch.Tensor, "edge_index": torch.Tensor})
ModelInput = TypedDict("ModelInput", {"x1": torch.Tensor, "y": torch.Tensor})
AllInput = TypedDict(
    "AllInput",
    {
        "x1": torch.Tensor,
        "y": torch.Tensor,
        "x2": torch.Tensor,
        "edge_index": torch.Tensor,
    },
)
