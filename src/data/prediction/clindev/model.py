"""
Trial characteristic and duration prediction model
"""
import torch
import torch.nn as nn
import logging

from data.prediction.clindev.types import TwoStageModelSizes

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
        _x = x.clone().view(self.orig_shape)  # clone?
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
        >>> model = TwoStageModel()
        >>> checkpoint = torch.load("clindev_model_checkpoints/checkpoint_15.pt", map_location=torch.device('mps'))
        >>> model.load_state_dict(checkpoint["model_state_dict"])
        >>> model.eval()
        >>> torch.save(model, 'clindev-model.pt')
    """

    def __init__(
        self,
        target_idxs: tuple[int, ...],
        original_shape: tuple[int, ...],
        sizes: TwoStageModelSizes,
    ):
        super().__init__()
        torch.device("mps")

        self.input_model = nn.ModuleDict(
            {
                "multi_select": nn.Linear(sizes.multi_select_input, sizes.stage1_input),
                "single_select": nn.Linear(
                    sizes.single_select_input, sizes.stage1_input
                ),
                "text": nn.Linear(sizes.text_input, sizes.stage1_input),
            }
        )
        # Stage 1 model
        self.stage1_model = nn.Sequential(
            nn.Linear(sizes.stage1_input, sizes.stage1_input),
            MaskLayer(target_idxs, original_shape),
            nn.Linear(sizes.stage1_input, sizes.stage1_hidden),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(sizes.stage1_hidden, sizes.stage1_embedded_output),
        )

        # used for calc of loss / evaluation of stage1 separately
        self.stage1_probs_model = nn.Sequential(
            nn.Linear(sizes.stage1_embedded_output, sizes.stage1_prob_output),
        )

        # Stage 2 model
        self.stage2_model = nn.Sequential(
            nn.Linear(
                sizes.stage2_input + sizes.stage1_embedded_output, sizes.stage2_hidden
            ),
            nn.ReLU(),
            nn.Linear(sizes.stage2_hidden, sizes.stage2_output),
        )

    @property
    def optimizer(self):
        return OPTIMIZER_CLASS(
            list(self.stage1_model.parameters())
            + list(self.stage1_probs_model.parameters())
            + list(self.stage2_model.parameters()),
            lr=LR,
        )

    def forward(
        self, multi_select_x, single_select_x, text_x
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.input_model["multi_select"].forward(multi_select_x)
        x = self.input_model["single_select"].forward(single_select_x)

        if text_x is not None:
            x = self.input_model["text"].forward(text_x)

        y1_pred = self.stage1_model(x)
        y1_probs = self.stage1_probs_model(y1_pred)
        x2 = torch.cat((x, y1_pred), dim=1)
        y2_pred = self.stage2_model(x2)
        return (y1_pred, y2_pred, y1_probs)
