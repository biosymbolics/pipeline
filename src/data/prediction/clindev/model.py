"""
Trial characteristic and duration prediction model
"""
import os
from typing import Literal, Optional
import torch
import torch.nn as nn
import logging

from data.prediction.clindev.utils import embed_cat_inputs

from .constants import (
    CHECKPOINT_PATH,
    DEVICE,
    LR,
    OPTIMIZER_CLASS,
)
from .types import StageSizes, TwoStageModelSizes, TwoStageModelInputSizes


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SaveableModel(nn.Module):
    def __init__(self, key: str, device: str):
        super().__init__()
        self.device = device
        self.key = key

    def get_checkpoint_name(self, epoch: int):
        return f"{self.key}-checkpoint_{epoch}.pt"

    def get_checkpoint_path(self, epoch: int):
        return os.path.join(CHECKPOINT_PATH, self.get_checkpoint_name(epoch))

    def save(self, epoch: int):
        """
        Save model checkpoint

        Args:
            epoch (int): Epoch number
        """
        checkpoint = {
            "model_state_dict": self.state_dict(),
            "epoch": epoch,
        }
        checkpoint_name = f"{self.key}-checkpoint_{epoch}.pt"

        try:
            torch.save(checkpoint, self.get_checkpoint_path(epoch))
            logger.info("Saved checkpoint %s", checkpoint_name)
        except Exception as e:
            logger.error("Failed to save checkpoint %s: %s", checkpoint_name, e)
            raise e

    def load(self, epoch: int):
        model = torch.load(
            self.get_checkpoint_path(epoch),
            map_location=torch.device(self.device),
        )
        model.eval()
        return model


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


class OutputCorrelationDecoders(SaveableModel, nn.ModuleDict):
    def __init__(
        self, output_map: Optional[dict[str, int]] = None, device: str = DEVICE
    ):
        super().__init__("correlation_decoders", device)
        if output_map is not None:
            for name in output_map.keys():
                self.add_module(
                    name, OutputCorrelation.create(name, output_map).to(device)
                )

            self.to(device)

    def forward(self, y1_probs_list):
        values = [
            model(*[y1_prob for i2, y1_prob in enumerate(y1_probs_list) if i2 != i])
            for i, model in enumerate(self.values())
        ]
        return torch.cat(values, dim=1)


class Stage1Output(SaveableModel, nn.ModuleDict):
    def __init__(
        self,
        input_size: Optional[int] = None,
        output_map: Optional[dict[str, int]] = None,
        device: str = DEVICE,
    ):
        super().__init__("stage1_output", device)
        if output_map is not None and input_size is not None:
            for name, size in output_map.items():
                self.add_module(name, nn.Linear(input_size, size))

            self.to(device)

    def forward(self, x):
        values = list([module(x) for module in self.values()])
        return torch.cat(values, dim=1).to(self.device), values


class InputModel(SaveableModel):
    def __init__(
        self, sizes: Optional[TwoStageModelInputSizes] = None, device: str = DEVICE
    ):
        super().__init__("input", device)

        if sizes is not None:
            logger.info("Initializing input model with sizes %s", sizes)
            self.multi_select_embeddings = nn.ModuleDict(
                {
                    f: nn.Embedding(s, sizes.embedding_dim)
                    for f, s in sizes.categories_by_field.multi_select.items()
                }
            )
            self.multi_select = nn.Linear(
                sizes.multi_select_input * sizes.embedding_dim,
                sizes.multi_select_output,
            )

            self.single_select_embeddings = nn.ModuleDict(
                {
                    f: nn.Embedding(s, sizes.embedding_dim)
                    for f, s in sizes.categories_by_field.single_select.items()
                }
            )
            self.single_select = nn.Linear(
                sizes.single_select_input * sizes.embedding_dim,
                sizes.single_select_output,
            )

            self.quantitative = nn.Linear(
                sizes.quantitative_input, sizes.quantitative_input
            )
            self.text = nn.Linear(sizes.text_input, sizes.text_input)

            self.to(device)

    def forward(self, multi_select, single_select, text, quantitative):
        all_inputs = []
        if len(text) > 0:
            all_inputs.append(self.text(text))

        if len(quantitative) > 0:
            all_inputs.append(self.quantitative(quantitative))

        if len(multi_select) > 0:
            inputs = embed_cat_inputs(
                multi_select, self.multi_select_embeddings, self.device
            )
            all_inputs.append(self.multi_select(inputs))

        if len(single_select) > 0:
            inputs = embed_cat_inputs(
                single_select, self.single_select_embeddings, self.device
            )
            all_inputs.append(self.single_select(inputs))

        x = torch.cat(all_inputs, dim=1)
        return x


class Stage1Model(SaveableModel):
    def __init__(self, sizes: Optional[StageSizes] = None, device: str = DEVICE):
        super().__init__("stage1", device)
        if sizes is not None:
            logger.info("Initializing input model with sizes %s", sizes)
            self.layer1 = nn.Sequential(
                nn.Linear(sizes.input, sizes.input),
                nn.Dropout(0.01),
                nn.BatchNorm1d(sizes.input),
                nn.ReLU(),
            ).to(device)

            self.layer2 = nn.Sequential(
                nn.Linear(sizes.input, sizes.hidden),
                nn.Dropout(0.2),
                nn.BatchNorm1d(sizes.hidden),
                nn.ReLU(),
            ).to(device)

            self.layer3 = nn.Sequential(
                nn.Linear(sizes.hidden, sizes.output),
            ).to(device)

    def forward(self, input):
        print("STAGE11 INPUT SHAPE", input.shape)
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Stage2Model(SaveableModel):
    def __init__(self, sizes: Optional[StageSizes] = None, device: str = DEVICE):
        super().__init__("stage2", device)

        if sizes is not None:
            self.layer1 = nn.Sequential(
                nn.Linear(sizes.input, sizes.hidden),
                nn.Dropout(0.01),
                nn.BatchNorm1d(sizes.hidden),
                nn.ReLU(),
            ).to(device)

            self.layer2 = nn.Sequential(
                nn.Linear(sizes.hidden, round(sizes.hidden / 2)),
                nn.Dropout(0.2),
                nn.BatchNorm1d(round(sizes.hidden / 2)),
                nn.ReLU(),
            ).to(device)

            self.layer3 = nn.Sequential(
                nn.Linear(round(sizes.hidden / 2), sizes.output),
                nn.Softmax(),
            ).to(device)

    def forward(self, input):
        print("STAGE2 INPUT SHAPE", input.shape)
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


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
        mode: Literal["train", "predict", "none"] = "train",
        sizes: Optional[TwoStageModelSizes] = None,  # required if mode == "train"
        checkpoint_epoch: Optional[int] = None,  # required if mode == "predict"
        device: str = DEVICE,
    ):
        super().__init__()
        torch.device(device)
        self.device = device

        if mode == "train" and sizes is not None:
            self.__initialize_model(sizes, device)
        elif mode == "predict" and checkpoint_epoch is not None:
            return self.__load_model(checkpoint_epoch)
        elif mode == "none":
            pass
        else:
            raise ValueError(
                f"Invalid mode: {mode}, checkpoint: {checkpoint_epoch}, sizes: {sizes}"
            )

    def save(self, epoch: int):
        """
        Save model checkpoint

        Args:
            epoch (int): Epoch number
        """
        self.input_model.save(epoch)
        self.stage1_model.save(epoch)
        self.stage1_output_model.save(epoch)
        self.correlation_decoders.save(epoch)
        self.stage2_model.save(epoch)

    def load(self, epoch: int):
        self.input_model = InputModel().load(epoch)
        self.stage1_model = Stage1Model().load(epoch)
        self.stage1_output_model = Stage1Output().load(epoch)
        self.correlation_decoders = OutputCorrelationDecoders().load(epoch)
        self.stage2_model = Stage2Model().load(epoch)

    def __initialize_model(self, sizes: TwoStageModelSizes, device: str):
        self.input_model = InputModel(sizes, device=device)

        self.stage1_model = Stage1Model(
            StageSizes(sizes.stage1_input, sizes.stage1_hidden, sizes.stage1_output),
            device=device,
        )

        # used for calc of loss / evaluation of stage1 separately
        self.stage1_output_model = Stage1Output(
            sizes.stage1_output, sizes.stage1_output_map, device
        )

        # used to learn correlations between outputs
        self.correlation_decoders = OutputCorrelationDecoders(
            sizes.stage1_output_map, device
        )

        self.stage2_model = Stage2Model(
            StageSizes(sizes.stage2_input, sizes.stage2_hidden, 1), device
        )

    @property
    def optimizer(self):
        return OPTIMIZER_CLASS(
            list(self.input_model.parameters())
            + list(self.correlation_decoders.parameters())
            + list(self.stage1_model.parameters())
            + list(self.stage1_output_model.parameters())
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
        x = self.input_model(multi_select, single_select, text, quantitative)
        y1_pred = self.stage1_model(x).to(self.device)
        y1_probs, y1_probs_list = self.stage1_output_model(y1_pred)

        # outputs guessed based on the other outputs (to learn relationships)
        y1_corr_probs = self.correlation_decoders(y1_probs_list)

        y2_pred = self.stage2_model(y1_pred).to(self.device)

        return (y1_probs, y1_corr_probs, y2_pred)


class ClindevPredictionModel(TwoStageModel):
    def __init__(self, checkpoint_epoch: int, device: str = DEVICE):
        super().__init__("predict", None, checkpoint_epoch, device=device)


class ClindevTrainingModel(TwoStageModel):
    def __init__(self, sizes: TwoStageModelSizes, device: str = DEVICE):
        super().__init__("train", sizes, device=device)
