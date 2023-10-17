"""
Trial characteristic and duration prediction model
"""
import os
from typing import Literal, Optional
import torch
import torch.nn as nn
import logging

from data.prediction.clindev.utils import embed_cat_inputs
from utils.model import od_to_dict

from .constants import (
    CHECKPOINT_PATH,
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


class Stage1Model(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, device: str
    ):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.Dropout(0.01),
            nn.BatchNorm1d(input_size),
            nn.ReLU(),
            nn.Linear(input_size, hidden_size),
            nn.Dropout(0.2),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size),
        ).to(device)


class Stage2Model(nn.Module):
    def __init__(
        self, input_size: int, hidden_size: int, output_size: int, device: str
    ):
        # Stage 2 model
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Dropout(0.01),
            nn.BatchNorm1d(hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, round(hidden_size / 2)),
            nn.Dropout(0.2),
            nn.BatchNorm1d(round(hidden_size / 2)),
            nn.ReLU(),
            nn.Linear(round(hidden_size / 2), output_size),
            nn.Softmax(),
        ).to(device)


class TwoStageModel(nn.Module):
    """
    Predicts characteristics of a clinical trial

    TODO:
    - enrich interventions with MoA (but may bias - those that have MoA mappings are those that are more likely to have been successful)
    - biobert for tokenization of conditions and interventions (tokens have meaning e.g. in biologic names)
    - constrastive learning for stage 1
    """

    def __init__(
        self,
        mode: Literal["train", "predict", "none"],
        sizes: Optional[TwoStageModelSizes] = None,  # required if mode == "train"
        checkpoint: Optional[str] = None,  # required if mode == "predict"
        device: str = DEVICE,
    ):
        super().__init__()
        torch.device(device)
        self.device = device

        if mode == "train" and sizes is not None:
            self.__initialize_model(sizes, device)
        elif mode == "predict" and checkpoint is not None:
            return self.__load_model(checkpoint, device)
        elif mode == "none":
            pass
        else:
            raise ValueError(
                f"Invalid mode: {mode}, checkpoint: {checkpoint}, sizes: {sizes}"
            )

    def save_checkpoint(self, epoch: int):
        """
        Save model checkpoint

        Args:
            epoch (int): Epoch number
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "epoch": epoch,
        }
        checkpoint_name = f"checkpoint_{epoch}.pt"

        try:
            torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, checkpoint_name))
            logger.info("Saved checkpoint %s", checkpoint_name)
        except Exception as e:
            logger.error("Failed to save checkpoint %s: %s", checkpoint_name, e)
            raise e

    def __load_model(self, checkpoint: str, device: str):
        checkpoint_obj = torch.load(
            f"{CHECKPOINT_PATH}/{checkpoint}",
            map_location=torch.device(device),
        )

        state = od_to_dict(checkpoint_obj["model_state_dict"])

        def create_and_load(LayerOrModel, state, f, is_reversed: bool = True):
            print("WIWIWIW", f, state.keys())

            if state.get("weight") is not None:
                shape = state.get("weight").shape

                if is_reversed:
                    shape = tuple(reversed(shape))
                layer = LayerOrModel(*shape).to(device)
                layer.load_state_dict(state)
                return layer

            if state.get("0") is not None:
                print("0000keys", state.get("0").keys())
                model = LayerOrModel(
                    state["0"].weight.size(0),
                    state["4"].weight.size(0),
                    state["8"].weight.size(1),
                ).to(device)
                model.load_state_dict(state)
                return model

            raise ValueError(f"Invalid state: {state}")

        self.input_model = nn.ModuleDict(
            {
                f: create_and_load(nn.Linear, v, f)
                for f, v in state["input_model"].items()
            }
        )
        self.multi_select_embeddings = nn.ModuleDict(
            {
                f: create_and_load(nn.Embedding, v, f, is_reversed=False)
                for f, v in state["multi_select_embeddings"].items()
            }
        )
        self.single_select_embeddings = nn.ModuleDict(
            {
                f: create_and_load(nn.Embedding, v, f, is_reversed=False)
                for f, v in state["single_select_embeddings"].items()
            }
        )
        self.stage1_model = create_and_load(
            Stage1Model, state["stage1_model"], "stage1"
        )
        self.stage2_model = create_and_load(
            Stage2Model, state["stage2_model"], "stage2"
        )

        self.eval()

    def __initialize_model(self, sizes: TwoStageModelSizes, device: str):
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
        self.stage1_model = Stage1Model(
            sizes.stage1_input, sizes.stage1_hidden, sizes.stage1_output, device=device
        )

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

        self.stage2_model = Stage2Model(
            sizes.stage2_input, sizes.stage2_hidden, 1, device
        )

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


class ClindevPredictionModel(TwoStageModel):
    def __init__(self, checkpoint: str, device: str = DEVICE):
        super().__init__("predict", None, checkpoint, device=device)


class ClindevTrainingModel(TwoStageModel):
    def __init__(self, sizes: TwoStageModelSizes, device: str = DEVICE):
        super().__init__("train", sizes, device=device)
