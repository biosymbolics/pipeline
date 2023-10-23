"""
Trial characteristic and duration prediction model
"""
from typing import Literal, Optional
import torch
import torch.nn as nn
import logging

from data.prediction.clindev.utils import embed_cat_inputs
from data.prediction.model import SaveableModel

from .constants import (
    CHECKPOINT_PATH,
    DEVICE,
    LR,
    OPTIMIZER_CLASS,
)
from .types import StageSizes, ClinDevModelSizes, ClinDevModelInputSizes


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


class OutputCorrelationDecoders(SaveableModel, nn.ModuleDict):
    key = "correlation_decoders"
    device = DEVICE
    checkpoint_path = CHECKPOINT_PATH

    def __init__(self, output_map: Optional[dict[str, int]] = None):
        super().__init__()
        if output_map is not None:
            for name in output_map.keys():
                self.add_module(name, OutputCorrelation.create(name, output_map))

            self.to(self.device)

    def forward(self, y1_probs_list):
        values = [
            model(*[y1_prob for i2, y1_prob in enumerate(y1_probs_list) if i2 != i])
            for i, model in enumerate(self.values())
        ]
        return torch.cat(values, dim=1)


class Stage1Output(SaveableModel, nn.ModuleDict):
    key = "stage1_output"
    device = DEVICE
    checkpoint_path = CHECKPOINT_PATH

    def __init__(
        self,
        input_size: Optional[int] = None,
        output_map: Optional[dict[str, int]] = None,
    ):
        super().__init__()
        if output_map is not None and input_size is not None:
            for name, size in output_map.items():
                self.add_module(name, nn.Linear(input_size, size))

            self.to(self.device)

    def forward(self, x):
        values = list([module(x) for module in self.values()])
        return torch.cat(values, dim=1).to(self.device), values


class InputModel(SaveableModel):
    key = "input"
    device = DEVICE
    checkpoint_path = CHECKPOINT_PATH

    def __init__(self, sizes: Optional[ClinDevModelInputSizes] = None):
        super().__init__()

        if sizes is not None:
            logger.info("Initializing input model with sizes %s", sizes)

            self.multi_select_embeddings = (
                nn.ModuleDict(
                    {
                        f: nn.Embedding(s, sizes.embedding_dim)
                        for f, s in sizes.categories_by_field.multi_select.items()
                    }
                )
                if sizes.categories_by_field.multi_select is not None
                else None
            )
            self.multi_select = nn.Linear(
                sizes.multi_select_input * sizes.embedding_dim,
                sizes.multi_select_output,
            )

            self.single_select_embeddings = (
                nn.ModuleDict(
                    {
                        f: nn.Embedding(s, sizes.embedding_dim)
                        for f, s in sizes.categories_by_field.single_select.items()
                    }
                )
                if sizes.categories_by_field.single_select is not None
                else None
            )
            self.single_select = nn.Linear(
                sizes.single_select_input * sizes.embedding_dim,
                sizes.single_select_output,
            )

            self.quantitative = nn.Linear(
                sizes.quantitative_input, sizes.quantitative_input
            )
            self.text = nn.Linear(sizes.text_input, sizes.text_input)

            self.to(self.device)

    def forward(
        self,
        multi_select: list[torch.Tensor],
        single_select: list[torch.Tensor],
        text: torch.Tensor,
        quantitative: torch.Tensor,
    ):
        all_inputs = []
        if len(text) > 0:
            all_inputs.append(self.text(text))

        if len(quantitative) > 0:
            all_inputs.append(self.quantitative(quantitative))

        if len(multi_select) > 0 and self.multi_select_embeddings is not None:
            inputs = embed_cat_inputs(
                multi_select, self.multi_select_embeddings, self.device
            )
            all_inputs.append(self.multi_select(inputs))

        if len(single_select) > 0 and self.single_select_embeddings is not None:
            inputs = embed_cat_inputs(
                single_select, self.single_select_embeddings, self.device
            )
            all_inputs.append(self.single_select(inputs))

        x = torch.cat(all_inputs, dim=1)
        return x

    def eval(self):
        super().eval()
        for emb in [
            *(self.multi_select_embeddings or {}).values(),
            *(self.single_select_embeddings or {}).values(),
        ]:
            emb.eval()


class Stage1Model(SaveableModel):
    key = "stage1"
    device = DEVICE
    checkpoint_path = CHECKPOINT_PATH

    def __init__(self, sizes: Optional[StageSizes] = None):
        super().__init__()
        if sizes is not None:
            logger.info("Initializing stage1 model with sizes %s", sizes)
            self.layer1 = nn.Sequential(
                nn.Linear(sizes.input, sizes.input),
                nn.Dropout(0.01),
                nn.BatchNorm1d(sizes.input),
                nn.ReLU(),
            )

            self.layer2 = nn.Sequential(
                nn.Linear(sizes.input, sizes.hidden),
                nn.Dropout(0.2),
                nn.BatchNorm1d(sizes.hidden),
                nn.ReLU(),
            )

            self.layer3 = nn.Sequential(
                nn.Linear(sizes.hidden, sizes.output),
            )

            self.to(self.device)

    def forward(self, input):
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class Stage2Model(SaveableModel):
    key = "stage2"
    device = DEVICE
    checkpoint_path = CHECKPOINT_PATH

    def __init__(self, sizes: Optional[StageSizes] = None):
        super().__init__()

        if sizes is not None:
            logger.info("Initializing stage2 model with sizes %s", sizes)
            self.layer1 = nn.Sequential(
                nn.Linear(sizes.input, sizes.hidden),
                nn.Dropout(0.01),
                nn.BatchNorm1d(sizes.hidden),
                nn.ReLU(),
            )

            self.layer2 = nn.Sequential(
                nn.Linear(sizes.hidden, round(sizes.hidden / 2)),
                nn.Dropout(0.2),
                nn.BatchNorm1d(round(sizes.hidden / 2)),
                nn.ReLU(),
            )

            self.layer3 = nn.Sequential(
                nn.Linear(round(sizes.hidden / 2), sizes.output),
            )

            self.to(self.device)

    def forward(self, y1, x):
        input = torch.cat([y1, x], dim=1)
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


class ClinDevModel(nn.Module):
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
        sizes: Optional[ClinDevModelSizes] = None,  # required if mode == "train"
        checkpoint_epoch: Optional[int] = None,  # required if mode == "predict"
        device: str = DEVICE,
    ):
        super().__init__()
        torch.device(device)
        self.device = device

        if mode == "train" and sizes is not None:
            self.__initialize_model(sizes)
        elif mode == "predict" and checkpoint_epoch is not None:
            return self.load(checkpoint_epoch)
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
        self.input_model = InputModel.load(epoch)
        self.stage1_model = Stage1Model.load(epoch)
        self.stage1_output_model = Stage1Output.load(epoch)
        self.correlation_decoders = OutputCorrelationDecoders.load(epoch)
        self.stage2_model = Stage2Model.load(epoch)

    def __initialize_model(self, sizes: ClinDevModelSizes):
        self.input_model = InputModel(sizes)

        self.stage1_model = Stage1Model(
            StageSizes(sizes.stage1_input, sizes.stage1_hidden, sizes.stage1_output),
        )

        # used for calc of loss / evaluation of stage1 separately
        self.stage1_output_model = Stage1Output(
            sizes.stage1_output, sizes.stage1_output_map
        )

        # used to learn correlations between outputs
        self.correlation_decoders = OutputCorrelationDecoders(sizes.stage1_output_map)

        self.stage2_model = Stage2Model(
            StageSizes(sizes.stage2_input, sizes.stage2_hidden, sizes.stage2_output),
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Note to self: if not training, first check if weights are updating
        print(list(self.stage2_model.modules())[0].layer3[0].weight)
        print(self.stage1_model[0].weight)
        print(list(self.multi_select_embeddings.values())[0].weight[0:10])

        """
        x = self.input_model(multi_select, single_select, text, quantitative)
        y1_pred = self.stage1_model(x).to(self.device)
        y1_probs, y1_probs_list = self.stage1_output_model(y1_pred)

        # outputs guessed based on the other outputs (to learn relationships)
        y1_corr_probs = self.correlation_decoders(y1_probs_list)

        y2_pred = self.stage2_model(y1_pred, x).to(self.device)

        return (y1_probs, y1_corr_probs, y2_pred, y1_probs_list)


class ClindevPredictionModel(ClinDevModel):
    def __init__(self, checkpoint_epoch: int, device: str = DEVICE):
        super().__init__("predict", None, checkpoint_epoch, device=device)


class ClindevTrainingModel(ClinDevModel):
    def __init__(self, sizes: ClinDevModelSizes, device: str = DEVICE):
        super().__init__("train", sizes, device=device)
