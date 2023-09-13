"""
Trial characteristic and duration prediction model
"""
import torch
import torch.nn as nn
import logging

from .constants import (
    LR,
    OPTIMIZER_CLASS,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class MaskLayer(nn.Module):
    def __init__(self, target_idxs: tuple[int, ...]):
        super().__init__()
        self.target_idxs = target_idxs

    def forward(self, x):
        mask = torch.ones_like(x)  # tensor of 1s
        mask[:, ~torch.tensor(self.target_idxs)] = 0  # mask out non-target indices
        return x * mask


class TwoStageModel(nn.Module):
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

    def __init__(
        self,
        target_idxs: tuple[int, ...],
        input_size: int,
        stage1_hidden_size: int = 64,
        stage1_output_size: int = 32,
        stage2_hidden_size: int = 64,
    ):
        torch.device("mps")

        super().__init__()

        # Stage 1 model
        self.stage1_model = nn.Sequential(
            nn.Linear(input_size, stage1_hidden_size),
            nn.Linear(stage1_hidden_size, stage1_hidden_size),  # mask layer
            MaskLayer(target_idxs),
            nn.ReLU(),
            nn.Linear(stage1_hidden_size, stage1_output_size),
        )

        # Stage 2 model
        self.stage2_model = nn.Sequential(
            nn.Linear(input_size + stage1_output_size, stage2_hidden_size),
            nn.ReLU(),
            nn.Linear(stage2_hidden_size, 1),  # Output size of 1 (duration)
        )

    @property
    def stage1_optimizer(self):
        return OPTIMIZER_CLASS(self.stage1_model.parameters(), lr=LR)

    @property
    def stage2_optimizer(self):
        return OPTIMIZER_CLASS(self.stage2_model.parameters(), lr=LR)

    def forward(self, x):
        print("X shape", x.size())
        y1_pred = self.stage1_model(x)  # Stage 1 inference
        print("X1 pred", y1_pred.size())
        x2 = torch.cat((x, y1_pred), dim=1)
        print("X2 cat", x2.size())
        y2_pred = self.stage2_model(x2)  # Stage 2 inference
        return (y1_pred, y2_pred)
