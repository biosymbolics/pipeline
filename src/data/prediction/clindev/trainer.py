import asyncio
import json
import logging
import math
import random
import sys
from typing import Any, Callable, NamedTuple, Optional, Sequence
from pydash import flatten
import torch
import torch.nn as nn
from ignite.metrics import Accuracy, MeanAbsoluteError, Precision, Recall
import polars as pl
from prisma.enums import (
    BiomedicalEntityType,
    ComparisonType,
    OwnerType,
    TrialDesign,
    TrialMasking,
    TrialPhase,
    TrialPurpose,
    TrialRandomization,
    TrialStatus,
)

import system
from typings.documents.trials import ScoredTrial

system.initialize()

from data.prediction.utils import (
    ModelInputAndOutput,
    decode_output,
    split_train_and_test,
)
from clients.documents.trials import client as trial_client
from clients.low_level.prisma import prisma_context
from utils.list import batch
from utils.encoding.json_encoder import DataclassJSONEncoder

from .constants import (
    ALL_FIELD_LISTS,
    BASE_ENCODER_DIRECTORY,
    BATCH_SIZE,
    DEVICE,
    EMBEDDING_DIM,
    SAVE_FREQUENCY,
    TRAINING_PROPORTION,
    InputAndOutputRecord,
    field_lists,
    output_field_lists,
)
from .model import ClindevTrainingModel
from .types import ClinDevModelSizes
from .utils import (
    calc_categories_loss,
    get_batch,
    prepare_data,
    preprocess_inputs,
    split_categories,
)

from ..types import AllCategorySizes, ModelInput


class MetricWrapper(NamedTuple):
    metric: Any
    transform: Optional[Callable]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

STAGE2_MSE_WEIGHT = 20
STAGE1_CORR_WEIGHT = 0.2


async def fetch_training_trials(
    status: TrialStatus, limit: int = 2000
) -> list[ScoredTrial]:
    """
    Fetch all trial summaries by status
    """
    async with prisma_context(600):
        trials = await trial_client.find_many(
            where={
                "status": status,
                "duration": {"gt": 0},
                "enrollment": {"gt": 0},
                "intervention_type": BiomedicalEntityType.PHARMACOLOGICAL,
                "interventions": {"some": {"id": {"gt": 0}}},
                "indications": {"some": {"id": {"gt": 0}}},
                "max_timeframe": {"gt": 0},
                "purpose": {
                    "in": [
                        TrialPurpose.TREATMENT.name,
                        TrialPurpose.BASIC_SCIENCE.name,
                    ]
                },
                # "sponsor": {"is_not": {"owner_type": OwnerType.OTHER}},
                "NOT": [
                    {"design": TrialDesign.UNKNOWN},
                    {"design": TrialDesign.FACTORIAL},
                    {"comparison_type": ComparisonType.UNKNOWN},
                    {"comparison_type": ComparisonType.OTHER},
                    {"masking": TrialMasking.UNKNOWN},
                    {"randomization": TrialRandomization.UNKNOWN},
                    {"phase": TrialPhase.NA},
                ],
            },
            include={
                "interventions": True,
                "indications": True,
                "outcomes": True,
                "sponsor": True,
            },
            take=limit,
        )

    logger.info("Fetched %s trials", len(trials))

    return trials


class ModelTrainer:
    """
    Trainable model
    """

    def __init__(
        self,
        training_input: ModelInputAndOutput,
        test_input: ModelInputAndOutput,
        category_sizes: AllCategorySizes,
        test_trials: Sequence[Sequence[dict]],
        embedding_dim: int = EMBEDDING_DIM,
    ):
        """
        Initialize model

        Args:
            inputs (ModelInputAndOutput): Input dict
            category_sizes (AllCategorySizes): Sizes of categorical fields
            trials (Sequence[dict]): Trials (for output decoding)
            embedding_dim (int, optional): Embedding dimension. Defaults to 16.
        """
        torch.device(DEVICE)
        self.device = DEVICE

        self.test_trials = test_trials  # used for decoding

        self.y1_category_sizes = category_sizes.y1
        self.training_input = training_input
        self.test_input = test_input

        sizes = ClinDevModelSizes(
            categories_by_field=category_sizes,
            embedding_dim=embedding_dim,
            multi_select_input=math.prod(training_input.multi_select.shape[2:]),
            quantitative_input=training_input.quantitative.size(-1),
            single_select_input=math.prod(training_input.single_select.shape[2:]),
            text_input=math.prod(training_input.text.shape[2:]),
            stage1_output_map=category_sizes.y1,
            stage1_output=math.prod(training_input.y1_true.shape[2:]) * 20,
            stage2_output=category_sizes.y2,
        )
        logger.info("Model sizes: %s", sizes)

        self.model = ClindevTrainingModel(sizes)
        self.stage1_criterion = nn.CrossEntropyLoss(label_smoothing=0.005)
        self.stage2_ce_criterion = nn.CrossEntropyLoss(label_smoothing=0.005)
        self.stage2_mse_criterion = nn.MSELoss()

        self.stage1_metrics = {
            # "cp": {k: ClassificationReport() for k in category_sizes.y1.keys()},
            "accuracy": {k: Accuracy() for k in category_sizes.y1.keys()},
        }

        self.stage2_metrics = {
            "precision": MetricWrapper(
                metric=Precision(), transform=lambda x: (x > 0.49).float()
            ),
            "recall": MetricWrapper(
                metric=Recall(), transform=lambda x: (x > 0.49).float()
            ),
            "mae": MetricWrapper(
                metric=MeanAbsoluteError(),
                transform=lambda x: torch.argmax(x, dim=1),
            ),
        }

    def __call__(self, *args, **kwargs):
        """
        Alias for self.train
        """
        self.train(*args, **kwargs)

    def calc_loss(
        self,
        i: int,
        batch: ModelInputAndOutput,
        y1_probs: torch.Tensor,
        y1_corr_probs: torch.Tensor,
        y2_preds: torch.Tensor,
    ) -> torch.Tensor:
        # STAGE 1 - categorical loss
        y1_probs_by_field, y1_true_by_field = split_categories(
            y1_probs, batch.y1_true, self.y1_category_sizes
        )
        stage1_loss = calc_categories_loss(
            y1_probs_by_field,
            y1_true_by_field,
            self.stage1_criterion,
        )

        # STAGE 1 - correlational loss (guessing vals based on peer outputs)
        y1_corr_probs_by_field, _ = split_categories(
            y1_corr_probs, batch.y1_true, self.y1_category_sizes
        )
        stage1_corr_loss = torch.mul(
            calc_categories_loss(
                y1_corr_probs_by_field,
                y1_true_by_field,
                self.stage1_criterion,
            ),
            STAGE1_CORR_WEIGHT,
        )

        # STAGE 2
        stage2_ce_loss = self.stage2_ce_criterion(y2_preds, batch.y2_oh_true)

        # using softmax to get a more precise error than CEL between one-hot vectors
        stage2_mse_loss = torch.mul(
            self.stage2_mse_criterion(torch.softmax(y2_preds, dim=1), batch.y2_oh_true),
            STAGE2_MSE_WEIGHT,
        )

        # TOTAL loss
        loss = stage1_loss + stage1_corr_loss + stage2_ce_loss + stage2_mse_loss

        logger.debug(
            "Batch %s Loss %s (Stage1 loss: %s (%s), Stage2: %s (%s))",
            i,
            loss.detach().item(),
            stage1_loss.detach().item(),
            stage1_corr_loss.detach().item(),
            stage2_ce_loss.detach().item(),
            stage2_mse_loss.detach().item(),
        )

        self.calculate_metrics(
            y1_probs_by_field, y1_true_by_field, y2_preds, batch.y2_oh_true
        )

        return loss

    def __train_batch(self, i: int, input: ModelInputAndOutput):
        """
        Train model on a single batch

        Args:
            i (int): Batch index
            num_batches (int): Number of batches
            input (ModelInputAndOutput): Input dict
        """
        batch = get_batch(i, input)

        if batch is None:
            return

        # place before any loss calculation
        self.model.optimizer.zero_grad()

        y1_probs, y1_corr_probs, y2_preds, _ = self.model(
            ModelInput.get_instance(**batch.__dict__)
        )

        loss = self.calc_loss(i, batch, y1_probs, y1_corr_probs, y2_preds)

        loss.backward()
        self.model.optimizer.step()

    def train(
        self,
        start_epoch: int = 0,
        num_epochs: int = 250,
    ):
        """
        Train model

        Args:
            start_epoch (int, optional): Epoch to start training from. Defaults to 0.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 20.
        """
        num_batches = self.training_input.single_select.size(0)

        for epoch in range(start_epoch, num_epochs):
            logger.info("Starting epoch %s", epoch)
            _input_dict = ModelInputAndOutput(
                **{
                    k: v.detach().clone()
                    for k, v in self.training_input.__dict__.items()
                },
            )
            for i in range(num_batches):
                logger.debug("Starting batch %s out of %s", i, num_batches)
                self.__train_batch(i, _input_dict)

            if epoch % SAVE_FREQUENCY == 0:
                self.log_metrics("Training")
                # TODO: unreliable
                num_eval_batches = self.test_input.single_select.size(0)
                for te in range(0, num_eval_batches - 1):
                    batch = get_batch(te, self.test_input)

                    if batch is None:
                        continue

                    self.evaluate(batch, self.test_trials[te], self.y1_category_sizes)

                self.log_metrics("Evaluation")
                self.model.save(epoch)

    def log_metrics(self, stage: str = "Training"):
        """
        Log metrics
        """
        try:
            for k in self.y1_category_sizes.keys():
                for name, metric in self.stage1_metrics.items():
                    value = metric[k].compute().__round__(2)
                    logger.info("%s Stage1 %s %s: %s", stage, k, name, value)
                    metric[k].reset()

            for name in self.stage2_metrics.keys():
                metric, _ = self.stage2_metrics[name]
                value = metric.compute()  # type: ignore
                if isinstance(value, torch.Tensor):
                    value = value.item()
                logger.info("%s Stage2 %s: %s", stage, name, value.__round__(2))
                metric.reset()  # type: ignore

        except Exception as e:
            logger.warning("Failed to evaluate: %s", e)

    def calculate_metrics(
        self,
        y1_preds_by_field: Sequence[torch.Tensor],
        y1_true_by_field: Sequence[torch.Tensor],
        y2_preds: torch.Tensor,
        y2_oh_true: torch.Tensor,
    ):
        """
        Calculate discrete metrics for a batch
        (Conversion to CPU because ignite has prob with MPS)
        """
        preds_by_trues = zip(y1_preds_by_field, y1_true_by_field)

        for i, (y1_preds, y1_true) in enumerate(preds_by_trues):
            k = list(self.y1_category_sizes.keys())[i]

            cpu_y1_preds = y1_preds.detach().to("cpu")
            cpu_y1_true = y1_true.detach().to("cpu")

            for metric in self.stage1_metrics.values():
                metric[k].update((cpu_y1_preds, cpu_y1_true))

        cpu_y2_preds = torch.softmax(y2_preds.detach().to("cpu"), dim=1)
        cpu_y2_oh_true = y2_oh_true.detach().to("cpu")
        for mw in self.stage2_metrics.values():
            if mw.transform is not None:
                mw.metric.update(
                    (mw.transform(cpu_y2_preds), mw.transform(cpu_y2_oh_true))
                )
            else:
                mw.metric.update((cpu_y2_preds, cpu_y2_oh_true))

    def evaluate(
        self,
        batch: ModelInputAndOutput,
        records: Sequence[dict],
        category_sizes: dict[str, int],
    ):
        """
        Evaluate model on eval/test set
        """

        y1_probs, _, y2_preds, _ = self.model(ModelInput.get_instance(**batch.__dict__))

        y1_probs_by_field, y1_true_by_field = split_categories(
            y1_probs, batch.y1_true, category_sizes
        )

        self.calculate_metrics(
            y1_probs_by_field, y1_true_by_field, y2_preds, batch.y2_oh_true
        )

        # decode outputs and verify a few
        outputs = decode_output(
            [y1.detach().to("cpu") for y1 in y1_probs_by_field],
            y2_preds.detach().to("cpu"),
            output_field_lists,
            directory=BASE_ENCODER_DIRECTORY,
        )
        print_fields = [
            "nct_id",
            "conditions",
            "phase",
            *flatten(output_field_lists.__dict__.values()),
        ]
        comparison = [
            {
                k: f"{v} (pred: {pred[k]})" if pred.get(k) is not None else v
                for k, v in true.items()
                if k in print_fields
            }
            for true, pred in zip(records, pl.DataFrame(outputs).to_dicts())
        ]
        logger.info(
            "Comparisons: %s",
            json.dumps(
                sorted(comparison, key=lambda x: random.random())[0:1],
                cls=DataclassJSONEncoder,
                indent=2,
            ),
        )

    @staticmethod
    async def train_from_trials(batch_size: int = BATCH_SIZE):
        trials = await fetch_training_trials(TrialStatus.COMPLETED, limit=50000)
        trials = preprocess_inputs(trials)
        trial_dicts = [t.__dict__ for t in trials]

        inputs, category_sizes = prepare_data(
            [
                InputAndOutputRecord(
                    **{k: v for k, v in t.items() if k in ALL_FIELD_LISTS}
                )
                for t in trial_dicts
            ],
            field_lists,
            batch_size=batch_size,
            device=DEVICE,
        )
        batched_trials = batch(trial_dicts, batch_size)

        training_input, test_input, _, test_records = split_train_and_test(
            inputs, batched_trials, TRAINING_PROPORTION
        )

        model = ModelTrainer(training_input, test_input, category_sizes, test_records)
        model.train()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.prediction.clindev.trainer
            """
        )
        sys.exit()

    asyncio.run(ModelTrainer.train_from_trials())
