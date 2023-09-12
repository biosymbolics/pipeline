"""
Patent Probability of Success (PoS) model(s)
"""
from typing import Any, Optional
import torch
import torch.nn as nn

OUTPUT_SIZE = 8


class TrialCharacteristicsModel(nn.Module):
    """
    Predicts characteristics of a clinical trial

    Loading:
        >>> import torch; import system; system.initialize();
        >>> from core.models.clindev.core import DNN
        >>> model = DNN()
        >>> checkpoint = torch.load("clindev_model_checkpoints/checkpoint_15.pt", map_location=torch.device('mps'))
        >>> model.load_state_dict(checkpoint["model_state_dict"])
        >>> model.eval()
        >>> torch.save(model, 'clindev-model.pt')
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_size: int = OUTPUT_SIZE):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dnn(x)

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return super().__call__(*args, **kwds)


class TrialDurationModel(nn.Module):
    """
    Predicts duration of a clinical trial
    """

    def __init__(
        self, input_dim: int = OUTPUT_SIZE, hidden_dim: int = 32, output_size: int = 1
    ):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Linear(hidden_dim, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dnn(x)

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return super().__call__(*args, **kwds)
