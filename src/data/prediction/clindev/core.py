"""
Patent Probability of Success (PoS) model(s)
"""
from typing import Any, Optional
import torch
import torch.nn as nn


class DNN(nn.Module):
    """
    DNN for clindev characterization

    Loading:
        >>> import torch; import system; system.initialize();
        >>> from core.models.clindev.core import DNN
        >>> model = DNN()
        >>> checkpoint = torch.load("clindev_model_checkpoints/checkpoint_15.pt", map_location=torch.device('mps'))
        >>> model.load_state_dict(checkpoint["model_state_dict"])
        >>> model.eval()
        >>> torch.save(model, 'clindev-model.pt')
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return layers(x1).squeeze()
        return self.dnn(x)

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return super().__call__(*args, **kwds)
