"""
Trial characteristic and duration prediction model
"""
import torch
import torch.nn as nn
import logging

from data.prediction.clindev.utils import embed_cat_inputs


from .constants import (
    DEVICE,
    LR,
    OPTIMIZER_CLASS,
)
from .types import TwoStageModelSizes


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class OutputCorrelation(nn.Module):
    """
    Model to learn based on output correlation
    (design/randomization/masking are correlated)
    """

    def __init__(self, other_outputs_size, this_output_size):
        super().__init__()
        self.decoder = nn.Linear(other_outputs_size, this_output_size)

    def forward(self, *inputs):
        x = torch.cat(inputs, dim=1)
        return self.decoder(x)

    @staticmethod
    def create(output_field, stage1_output_map) -> "OutputCorrelation":
        hidden_size = sum(
            [s for f, s in stage1_output_map.items() if f != output_field]
        )
        return OutputCorrelation(hidden_size, stage1_output_map[output_field])


class TwoStageModel(nn.Module):
    """
    Predicts characteristics of a clinical trial

    TODO:
    - enrich interventions with MoA (but may bias - those that have MoA mappings are those that are more likely to have been successful)
    - biobert for tokenization of conditions and interventions (tokens have meaning e.g. in biologic names)
    - constrastive learning for stage 1

    Loading:
        >>> import torch; import system; system.initialize();
        >>> from core.models.clindev.model import TwoStageModel
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
            nn.Dropout(0.01),
            nn.BatchNorm1d(sizes.stage1_input),
            nn.ReLU(),
            nn.Linear(sizes.stage1_input, sizes.stage1_hidden),
            nn.Dropout(0.2),
            nn.BatchNorm1d(sizes.stage1_hidden),
            nn.ReLU(),
            nn.Linear(sizes.stage1_hidden, sizes.stage1_output),
        ).to(device)

        # used for calc of loss / evaluation of stage1 separately
        self.stage1_output_models = nn.ModuleDict(
            dict(
                [
                    (f, nn.Linear(sizes.stage1_output, size))
                    for f, size in sizes.stage1_output_map.items()
                ]
            )
        ).to(device)

        self.correlation_decoders = nn.ModuleDict(
            dict(
                [
                    (f, OutputCorrelation.create(f, sizes.stage1_output_map).to(device))
                    for f in sizes.stage1_output_map.keys()
                ]
            )
        )

        # Stage 2 model
        self.stage2_model = nn.Sequential(
            nn.Linear(sizes.stage1_output, sizes.stage2_hidden),
            nn.Dropout(0.01),
            nn.BatchNorm1d(sizes.stage2_hidden),
            nn.ReLU(),
            nn.Linear(sizes.stage2_hidden, round(sizes.stage2_hidden / 2)),
            nn.Dropout(0.2),
            nn.BatchNorm1d(round(sizes.stage2_hidden / 2)),
            nn.ReLU(),
            nn.Linear(round(sizes.stage2_hidden / 2), sizes.stage2_output),
            nn.Softmax(),
        ).to(device)

    @property
    def optimizer(self):
        return OPTIMIZER_CLASS(
            list(self.multi_select_embeddings.parameters())
            + list(self.single_select_embeddings.parameters())
            + list(self.correlation_decoders.parameters())
            + list(self.stage1_model.parameters())
            + list(self.stage1_output_models.parameters())
            + list(self.stage2_model.parameters()),
            lr=LR,
            weight_decay=1e-3,
        )

    def forward(
        self,
        multi_select: list[torch.Tensor],
        single_select: list[torch.Tensor],
        text: torch.Tensor,
        quantitative: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note to self: if not training, first check if weights are updating
        print(self.stage2_model[0].weight)
        print(self.stage1_model[0].weight)
        print(list(self.multi_select_embeddings.values())[0].weight[0:10])
        """

        all_inputs = []
        if len(text) > 0:
            all_inputs.append(self.input_model["text"](text))

        if len(quantitative) > 0:
            all_inputs.append(self.input_model["quantitative"](quantitative))

        if len(multi_select) > 0:
            inputs = embed_cat_inputs(
                multi_select, self.multi_select_embeddings, self.device
            )
            all_inputs.append(self.input_model["multi_select"](inputs))

        if len(single_select) > 0:
            inputs = embed_cat_inputs(
                single_select, self.single_select_embeddings, self.device
            )
            all_inputs.append(self.input_model["single_select"](inputs))

        x = torch.cat(all_inputs, dim=1).to(self.device)

        y1_pred = self.stage1_model(x).to(self.device)
        y1_probs_list = [model(y1_pred) for model in self.stage1_output_models.values()]
        y1_probs = torch.cat(y1_probs_list, dim=1).to(self.device)

        # outputs guessed based on the other outputs (to learn relationships)
        y1_corr_probs = torch.cat(
            [
                model(*[y1_prob for i2, y1_prob in enumerate(y1_probs_list) if i2 != i])
                for i, model in enumerate(self.correlation_decoders.values())
            ],
            dim=1,
        ).to(self.device)

        y2_pred = self.stage2_model(y1_pred).to(self.device)

        return (y1_probs, y1_corr_probs, y2_pred)
