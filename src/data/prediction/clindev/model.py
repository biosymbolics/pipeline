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

        input_layers = {
            k: v
            for k, v in [
                (
                    "multi_select",
                    nn.Linear(sizes.multi_select_input, sizes.stage1_input),
                ),
                (
                    "quantitative",
                    nn.Linear(sizes.quantitative_input, sizes.stage1_input),
                ),
                (
                    "single_select",
                    nn.Linear(sizes.single_select_input, sizes.stage1_input),
                ),
                ("text", nn.Linear(sizes.text_input, sizes.stage1_input)),
            ]
            if v.in_features > 0
        }
        self.input_model = nn.ModuleDict(input_layers).to("mps")

        # Stage 1 model
        self.stage1_model = nn.Sequential(
            nn.Linear(sizes.stage1_input, sizes.stage1_input),
            nn.Linear(sizes.stage1_input, sizes.stage1_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(sizes.stage1_hidden, sizes.stage1_embedded_output),
        ).to("mps")

        # used for calc of loss / evaluation of stage1 separately
        self.stage1_output_models = nn.ModuleDict(
            dict(
                [
                    (
                        field,
                        nn.Sequential(
                            nn.Linear(sizes.stage1_embedded_output, size),
                            # nn.Softmax(), # CEL does softmax
                        ),
                    )
                    for field, size in sizes.stage1_output_map.items()
                ]
            )
        ).to("mps")

        # Stage 2 model
        self.stage2_model = nn.Sequential(
            nn.Linear(sizes.stage1_embedded_output, sizes.stage2_hidden),
            nn.Linear(sizes.stage2_hidden, round(sizes.stage2_hidden / 2)),
            nn.Dropout(0.1),
            nn.Linear(round(sizes.stage2_hidden / 2), sizes.stage2_output),
        ).to("mps")

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
            weight_decay=1e-7,
        )

    def forward(
        self, multi_select_x, single_select_x, text_x, quantitative_x
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note to self: if not training, first check if weights are updating
        print(self.stage2_model[0].weight)
        print(self.stage1_model[0].weight)
        """

        x = self.input_model["multi_select"](multi_select_x)
        x = self.input_model["single_select"](single_select_x)

        if len(text_x) > 0:
            x = self.input_model["text"](text_x)

        if len(quantitative_x) > 0:
            x = self.input_model["quantitative"](quantitative_x)

        y1_pred = self.stage1_model(x)

        y_probs = torch.cat(
            [model(y1_pred) for model in self.stage1_output_models.values()],
            dim=1,
        )
        y2_pred = self.stage2_model(y1_pred)

        return (y1_pred, y2_pred, y_probs)
