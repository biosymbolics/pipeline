from functools import partial
import logging
import math
import os
import sys
from typing import Any, Callable, NamedTuple, Optional, Sequence, cast
import torch
import torch.nn as nn
from ignite.metrics import Accuracy, ClassificationReport, MeanAbsoluteError

import system

system.initialize()

from clients.trials import fetch_trials
from data.prediction.utils import ModelInput
from data.types import FieldLists

from .constants import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    DEVICE,
    EMBEDDING_DIM,
    MULTI_SELECT_CATEGORICAL_FIELDS,
    SINGLE_SELECT_CATEGORICAL_FIELDS,
    QUANTITATIVE_FIELDS,
    QUANTITATIVE_TO_CATEGORY_FIELDS,
    SAVE_FREQUENCY,
    TEXT_FIELDS,
    Y1_CATEGORICAL_FIELDS,
    Y2_FIELD,
)
from .model import TwoStageModel
from .types import AllCategorySizes, TwoStageModelSizes
from .utils import (
    calc_categories_loss,
    prepare_inputs,
    preprocess_inputs,
    split_categories,
    split_train_and_test,
)


class MetricWrapper(NamedTuple):
    metric: Any
    transform: Optional[Callable]


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelTrainer:
    """
    Trainable model
    """

    def __init__(
        self,
        input_dict: ModelInput,
        category_sizes: AllCategorySizes,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        """
        Initialize model

        Args:
            input_dict (ModelInput): Input dict
            category_sizes (AllCategorySizes): Sizes of categorical fields
            embedding_dim (int, optional): Embedding dimension. Defaults to 16.
        """
        torch.device(DEVICE)
        self.device = DEVICE

        self.y1_category_sizes = category_sizes.y1
        self.input_dict = input_dict

        training_input_dict, test_input_dict = split_train_and_test(input_dict)
        self.training_input_dict = training_input_dict
        self.test_input_dict = test_input_dict

        sizes = TwoStageModelSizes(
            categories_by_field=category_sizes,
            embedding_dim=embedding_dim,
            multi_select_input=math.prod(input_dict.multi_select.shape[2:]),
            quantitative_input=input_dict.quantitative.size(-1),
            single_select_input=math.prod(input_dict.single_select.shape[2:]),
            text_input=input_dict.text.size(-1),
            stage1_output_map=category_sizes.y1,
            stage1_output=math.prod(input_dict.y1_true.shape[2:]),
            stage2_output=10,  # math.prod(input_dict["y2"].shape[2:]), # should be num values
        )
        logger.info("Model sizes: %s", sizes)

        self.model = TwoStageModel(sizes)
        self.stage1_criterion = nn.CrossEntropyLoss(label_smoothing=0.005)
        self.stage2_criterion = nn.CrossEntropyLoss()

        self.stage1_metrics = {
            "cp": {k: ClassificationReport() for k in category_sizes.y1.keys()},
            "accuracy": {k: Accuracy() for k in category_sizes.y1.keys()},
        }

        self.stage2_metrics = {
            "cp": MetricWrapper(metric=ClassificationReport(), transform=None),
            "accuracy": MetricWrapper(metric=Accuracy(), transform=None),
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

    def save_checkpoint(self, epoch: int):
        """
        Save model checkpoint

        Args:
            model (CombinedModel): Model to save
            optimizer (torch.optim.Optimizer): Optimizer to save
            epoch (int): Epoch number
        """
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.model.optimizer.state_dict(),
            "epoch": epoch,
        }
        checkpoint_name = f"checkpoint_{epoch}.pt"

        try:
            torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, checkpoint_name))
            logger.info("Saved checkpoint %s", checkpoint_name)
        except Exception as e:
            logger.error("Failed to save checkpoint %s: %s", checkpoint_name, e)
            raise e

    def calc_loss(
        self,
        i: int,
        batch: ModelInput,
        y1_probs: torch.Tensor,
        y1_corr_probs: torch.Tensor,
        y2_preds: torch.Tensor,
    ) -> torch.Tensor:
        # STAGE 1
        # main stage 1 categorical loss
        y1_probs_by_field, y1_true_by_field = split_categories(
            y1_probs, batch.y1_true, self.y1_category_sizes
        )
        stage1_loss = calc_categories_loss(
            y1_probs_by_field,
            y1_true_by_field,
            self.stage1_criterion,
        )

        # corr stage 1 categorical loss (guessing vals based on peer outputs)
        y1_corr_probs_by_field, _ = split_categories(
            y1_corr_probs, batch.y1_true, self.y1_category_sizes
        )
        stage1_corr_loss = calc_categories_loss(
            y1_corr_probs_by_field,
            y1_true_by_field,
            self.stage1_criterion,
        )

        # STAGE 2
        stage2_loss = self.stage2_criterion(y2_preds, batch.y2_true)

        # TOTAL
        loss = stage1_loss + torch.mul(stage1_corr_loss, 0.1) + stage2_loss

        logger.debug(
            "Batch %s Loss %s (Stage1 loss: %s (%s), Stage2: %s)",
            i,
            loss.detach().item(),
            stage1_loss.detach().item(),
            stage1_corr_loss.detach().item(),
            stage2_loss.detach().item(),
        )

        self.calculate_metrics(
            y1_probs_by_field, y1_true_by_field, y2_preds, batch.y2_true
        )

        return loss

    @staticmethod
    def __get_batch(i: int, input_dict: ModelInput) -> ModelInput:
        """
        Get input_dict for batch i
        """
        batch = ModelInput(
            **{
                f: v[i] if len(v) > i else torch.Tensor()
                for f, v in input_dict._asdict().items()
                if v is not None
            }
        )
        return batch

    def __train_batch(self, i: int, input_dict: ModelInput):
        """
        Train model on a single batch

        Args:
            i (int): Batch index
            num_batches (int): Number of batches
            input_dict (ModelInput): Input dict
        """
        batch = ModelTrainer.__get_batch(i, input_dict)

        # place before any loss calculation
        self.model.optimizer.zero_grad()

        y1_probs, y1_corr_probs, y2_preds = self.model(
            torch.split(batch.multi_select, 1, dim=1),
            torch.split(batch.single_select, 1, dim=1),
            batch.text,
            batch.quantitative,
        )

        # TOTAL
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
        num_batches = self.training_input_dict.multi_select.size(0)

        for epoch in range(start_epoch, num_epochs):
            logger.info("Starting epoch %s", epoch)
            _input_dict = ModelInput(
                **{k: v.detach().clone() for k, v in self.input_dict._asdict().items()},
            )
            for i in range(num_batches):
                logger.info("Starting batch %s out of %s", i, num_batches)
                self.__train_batch(i, _input_dict)

            if epoch % SAVE_FREQUENCY == 0:
                self.log_metrics("Training")
                num_eval_batches = self.test_input_dict.multi_select.size(0)
                for te in range(0, num_eval_batches):
                    batch = ModelTrainer.__get_batch(te, self.test_input_dict)
                    self.evaluate(batch, self.y1_category_sizes)
                self.log_metrics("Evaluation")
                self.save_checkpoint(epoch)

    def calculate_metrics(
        self,
        y1_preds_by_field: Sequence[torch.Tensor],
        y1_true_by_field: Sequence[torch.Tensor],
        y2_preds: torch.Tensor,
        y2_true: torch.Tensor,
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

        cpu_y2_preds = y2_preds.detach().to("cpu")
        cpu_y2_true = y2_true.detach().to("cpu")
        for metric, transform in self.stage2_metrics.values():
            if transform:
                metric.update((transform(cpu_y2_preds), cpu_y2_true))
            else:
                metric.update((cpu_y2_preds, cpu_y2_true))

    def log_metrics(self, stage: str = "Training"):
        """
        Log metrics
        """
        try:
            for k in self.y1_category_sizes.keys():
                for name, metric in self.stage1_metrics.items():
                    logger.info(
                        "%s Stage1 %s %s: %s", stage, k, name, metric[k].compute()
                    )
                    metric[k].reset()

            for name in self.stage2_metrics.keys():
                metric, _ = self.stage2_metrics[name]
                logger.info("%s Stage2 %s: %s", stage, name, metric.compute())  # type: ignore
                metric.reset()  # type: ignore

        except Exception as e:
            logger.warning("Failed to evaluate: %s", e)

    def evaluate(self, input_dict: ModelInput, category_sizes: dict[str, int]):
        """
        Evaluate model on eval/test set
        """
        y1_probs, _, y2_preds = self.model(
            torch.split(input_dict.multi_select, 1, dim=1),
            torch.split(input_dict.single_select, 1, dim=1),
            input_dict.text,
            input_dict.quantitative,
        )

        y1_probs_by_field, y1_true_by_field = split_categories(
            y1_probs, input_dict.y1_true, category_sizes
        )

        self.calculate_metrics(
            y1_probs_by_field, y1_true_by_field, y2_preds, input_dict.y2_true
        )

    @staticmethod
    def train_from_trials(batch_size: int = BATCH_SIZE):
        trials = preprocess_inputs(
            fetch_trials("COMPLETED", limit=2000), QUANTITATIVE_TO_CATEGORY_FIELDS
        )

        field_lists = FieldLists(
            single_select=SINGLE_SELECT_CATEGORICAL_FIELDS,
            multi_select=MULTI_SELECT_CATEGORICAL_FIELDS,
            text=TEXT_FIELDS,
            quantitative=QUANTITATIVE_FIELDS,
            y1_categorical=Y1_CATEGORICAL_FIELDS,
            y2=Y2_FIELD,
        )

        input_dict, category_sizes, num_batches = prepare_inputs(
            trials, field_lists, batch_size, DEVICE
        )

        model = ModelTrainer(input_dict, category_sizes)
        model.train()


def main():
    ModelTrainer.train_from_trials()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.prediction.clindev.trainer
            """
        )
        sys.exit()

    main()
