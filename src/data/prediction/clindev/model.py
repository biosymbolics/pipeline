"""
Trial characteristic and duration prediction model
"""
import math
import torch
import torch.nn as nn
import logging


from .constants import (
    DEVICE,
    LR,
    OPTIMIZER_CLASS,
)
from .types import TwoStageModelSizes


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
        device: str = DEVICE,
    ):
        super().__init__()
        torch.device(device)
        self.device = device
        embedding_dim = sizes.embedding_dim

        self.multi_select_embeddings = nn.ModuleDict(
            {
                f: nn.Embedding(s, embedding_dim, device=self.device)
                for f, s in sizes.categories_by_field.multi_select.items()
            }
        )

        self.single_select_embeddings = nn.ModuleDict(
            {
                f: nn.Embedding(s, embedding_dim, device=self.device)
                for f, s in sizes.categories_by_field.single_select.items()
            }
        )

        input_layers = {
            k: v
            for k, v in [
                (
                    "multi_select",
                    nn.Linear(
                        sizes.multi_select_input * embedding_dim,
                        sizes.multi_select_output,
                    ),
                ),
                (
                    "quantitative",
                    nn.Linear(sizes.quantitative_input, sizes.quantitative_input)
                    if sizes.quantitative_input > 0
                    else None,
                ),
                (
                    "single_select",
                    nn.Linear(
                        sizes.single_select_input * embedding_dim,
                        sizes.single_select_output,
                    ),
                ),
                (
                    "text",
                    nn.Linear(sizes.text_input, sizes.text_input)
                    if sizes.text_input > 0
                    else None,
                ),
            ]
            if v is not None
        }
        self.input_model = nn.ModuleDict(input_layers).to(device)

        # Stage 1 model
        # TODO: make contrastive??
        self.stage1_model = nn.Sequential(
            nn.Linear(sizes.stage1_input, sizes.stage1_input),
            nn.Linear(sizes.stage1_input, sizes.stage1_hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(sizes.stage1_hidden, sizes.stage1_output),
        ).to(device)

        # used for calc of loss / evaluation of stage1 separately
        self.stage1_output_models = nn.ModuleDict(
            dict(
                [
                    (field, nn.Linear(sizes.stage1_output, size))
                    for field, size in sizes.stage1_output_map.items()
                ]
            )
        ).to(device)

        # Stage 2 model
        self.stage2_model = nn.Sequential(
            nn.Linear(sizes.stage1_output, sizes.stage2_hidden),
            nn.Linear(sizes.stage2_hidden, round(sizes.stage2_hidden / 2)),
            nn.Dropout(0.1),
            nn.Linear(round(sizes.stage2_hidden / 2), sizes.stage2_output),
        ).to(device)

    @property
    def optimizer(self):
        return OPTIMIZER_CLASS(
            list(self.multi_select_embeddings.parameters())
            + list(self.single_select_embeddings.parameters())
            + list(self.stage1_model.parameters())
            + list(self.stage1_output_models.parameters())
            + list(self.stage2_model.parameters()),
            lr=LR,
            weight_decay=1e-5,
        )

    def forward(
        self,
        multi_select_x: list[torch.Tensor],
        single_select_x: list[torch.Tensor],
        text_x: torch.Tensor,
        quantitative_x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note to self: if not training, first check if weights are updating
        print(self.stage2_model[0].weight)
        print(self.stage1_model[0].weight)
        """

        # print(list(self.multi_select_embeddings.values())[0].weight[0:10])

        all_inputs = []
        if len(text_x) > 0:
            all_inputs.append(self.input_model["text"](text_x))

        if len(quantitative_x) > 0:
            all_inputs.append(self.input_model["quantitative"](quantitative_x))

        if len(multi_select_x) > 0:
            ms_input = torch.cat(
                [
                    el(x)
                    for x, el in zip(
                        multi_select_x,
                        self.multi_select_embeddings.values(),
                    )
                ],
                dim=1,
            ).to(self.device)
            ms_input = ms_input.view(*ms_input.shape[0:1], -1)
            all_inputs.append(self.input_model["multi_select"](ms_input))

        if len(single_select_x) > 0:
            ss_input = torch.cat(
                [
                    el(x)
                    for x, el in zip(
                        single_select_x,
                        self.single_select_embeddings.values(),
                    )
                ],
                dim=1,
            ).to(self.device)
            ss_input = ss_input.view(*ss_input.shape[0:1], -1)
            all_inputs.append(self.input_model["single_select"](ss_input))

        x = torch.cat(all_inputs, dim=1).to(self.device)

        y1_pred = self.stage1_model(x).to(self.device)

        y_probs = torch.cat(
            [model(y1_pred) for model in self.stage1_output_models.values()],
            dim=1,
        ).to(self.device)
        y2_pred = self.stage2_model(y1_pred).to(self.device)

        return (y1_pred, y2_pred, y_probs)
