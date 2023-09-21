"""
Trial characteristic and duration prediction model
"""
from pydash import flatten
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
        sizes: TwoStageModelSizes,
    ):
        super().__init__()
        torch.device("mps")

        self.input_model = nn.ModuleDict(
            {
                "multi_select": nn.Linear(sizes.multi_select_input, sizes.stage1_input),
                "text": nn.Linear(sizes.text_input, sizes.stage1_input),
                "single_select": nn.Linear(
                    sizes.single_select_input, sizes.stage1_input
                ),
            }
        )

        # Stage 1 model
        self.stage1_model = nn.Sequential(
            nn.Linear(sizes.stage1_input, sizes.stage1_input),
            nn.Linear(sizes.stage1_input, sizes.stage1_hidden),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(sizes.stage1_hidden, sizes.stage1_embedded_output),
        )

        # used for calc of loss / evaluation of stage1 separately
        self.stage1_output_models = dict(
            [
                (
                    field,
                    nn.Sequential(
                        nn.Linear(sizes.stage1_embedded_output, size),
                        nn.Sigmoid(),
                    ),
                )
                for field, size in sizes.stage1_output_map.items()
            ]
        )

        # Stage 2 model
        self.stage2_model = nn.Sequential(
            nn.Linear(sizes.stage1_embedded_output, sizes.stage2_hidden),
            nn.ReLU(),
            nn.Linear(sizes.stage2_hidden, sizes.stage2_output),
        )

    @property
    def optimizer(self):
        return OPTIMIZER_CLASS(
            list(self.stage1_model.parameters())
            + flatten(
                [
                    list(model.parameters())
                    for model in self.stage1_output_models.values()
                ]
            )
            + list(self.stage2_model.parameters()),
            lr=LR,
        )

    def forward(
        self, multi_select_x, single_select_x, text_x
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        x = self.input_model["multi_select"].forward(multi_select_x)
        x = self.input_model["single_select"].forward(single_select_x)
        if text_x is not None:
            x = self.input_model["text"].forward(text_x)

        y1_pred = self.stage1_model(x)

        y1_probs_dict = dict(
            [
                (field, model(y1_pred))
                for field, model in self.stage1_output_models.items()
            ]
        )
        y_probs = torch.cat(list(y1_probs_dict.values()), dim=1)
        y2_pred = self.stage2_model(y1_pred)
        return (y1_pred, y2_pred, y_probs, y1_probs_dict)
