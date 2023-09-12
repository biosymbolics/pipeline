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
        input_size: int,
        stage1_hidden_size: int = 64,
        stage1_output_size: int = 32,
        stage2_hidden_size: int = 64,
    ):
        super().__init__()

        logger.info(
            "Stage 1 dims - input: %s, hidden: %s, output: %s",
            input_size,
            stage1_hidden_size,
            stage1_output_size,
        )
        # Stage 1 model
        self.stage1_model = nn.Sequential(
            nn.Linear(input_size, stage1_hidden_size),
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
        x1_pred = self.stage1_model(x)  # Stage 1 inference
        x2 = torch.cat((x, x1_pred.detach()), dim=1)
        x2_pred = self.stage2_model(x2)  # Stage 2 inference
        return (x1_pred, x2_pred)
