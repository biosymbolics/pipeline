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
    def __init__(self, target_idxs: tuple[int, ...], orig_shape: tuple[int, ...]):
        super().__init__()
        self.target_idxs = target_idxs
        self.orig_shape = orig_shape

    def forward(self, x):
        _x = x.clone().view(self.orig_shape)
        mask = torch.ones_like(_x)  # tensor of 1s
        mask[:, ~torch.tensor(self.target_idxs)] = 0  # mask out non-target indices
        masked = _x * mask
        return masked.view(x.shape)


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
        original_shape: tuple[int, ...],
        stage1_hidden_size: int = 64,
        stage1_output_size: int = 32,
        stage2_hidden_size: int = 64,
    ):
        super().__init__()
        torch.device("mps")

        # Stage 1 model
        self.stage1_model = nn.Sequential(
            nn.Linear(input_size, input_size),
            # nn.Dropout(0.1),
            MaskLayer(target_idxs, original_shape),
            nn.Linear(input_size, stage1_hidden_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(stage1_hidden_size, stage1_output_size),
        )

        # Stage 2 model
        self.stage2_model = nn.Sequential(
            nn.Linear(input_size + stage1_output_size, stage2_hidden_size),
            nn.ReLU(),
            nn.Linear(stage2_hidden_size, 1),  # Output size of 1 (duration)
        )

    @property
    def optimizer(self):
        return OPTIMIZER_CLASS(self.stage1_model.parameters(), lr=LR)

    def forward(self, x):
        y1_pred = self.stage1_model(x)  # Stage 1 inference
        x2 = torch.cat((x, y1_pred), dim=1)
        y2_pred = self.stage2_model(x2)  # Stage 2 inference
        return (y1_pred, y2_pred)
