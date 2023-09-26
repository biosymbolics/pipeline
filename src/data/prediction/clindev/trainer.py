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
from .types import AllCategorySizes, DnnInput, TwoStageModelSizes
from .utils import calc_categories_loss, prepare_inputs, preprocess_inputs


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
        input_dict: DnnInput,
        category_sizes: AllCategorySizes,
        embedding_dim: int = EMBEDDING_DIM,
    ):
        """
        Initialize model

        Args:
            input_dim (int): Input dimension for DNN
            category_sizes (AllCategorySizes): Sizes of categorical fields
        """
        torch.device(DEVICE)
        self.device = DEVICE

        self.category_sizes = category_sizes

        sizes = TwoStageModelSizes(
            categories_by_field=category_sizes,
            embedding_dim=embedding_dim,
            multi_select_input=math.prod(input_dict["multi_select_x"].shape[2:]),
            quantitative_input=input_dict["quantitative_x"].size(-1),
            single_select_input=math.prod(input_dict["single_select_x"].shape[2:]),
            text_input=input_dict["text_x"].size(-1),
            stage1_output_map=category_sizes.y1,
            stage1_output=math.prod(input_dict["y1"].shape[2:]),
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

    def train(
        self,
        input_dict: DnnInput,
        num_batches: int,
        start_epoch: int = 0,
        num_epochs: int = 100,
    ):
        """
        Train model

        Args:
            input_dict (DnnInput): Dictionary of input tensors
            start_epoch (int, optional): Epoch to start training from. Defaults to 0.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 20.
        """
        for epoch in range(start_epoch, num_epochs):
            logger.info("Starting epoch %s", epoch)
            _input_dict = {f: v.detach().clone() for f, v in input_dict.items()}  # type: ignore
            for i in range(num_batches):
                logger.debug("Starting batch %s out of %s", i, num_batches)
                batch: DnnInput = cast(
                    DnnInput,
                    {
                        f: v[i] if len(v) > i else torch.Tensor()
                        for f, v in _input_dict.items()
                        if v is not None
                    },
                )

                # place before any loss calculation
                self.model.optimizer.zero_grad()

                y1_true = batch["y1"]
                y2_true = batch["y2"].squeeze().int()

                y1_probs, y1_corr_probs, y2_preds = self.model(
                    torch.split(batch["multi_select_x"], 1, dim=1),
                    torch.split(batch["single_select_x"], 1, dim=1),
                    batch["text_x"],
                    batch["quantitative_x"],
                )

                # STAGE 1
                # main stage 1 categorical loss
                stage1_loss, y1_probs_by_field, y1_true_by_field = calc_categories_loss(
                    y1_probs, y1_true, self.category_sizes.y1, self.stage1_criterion
                )

                # corr stage 1 categorical loss (guessing vals based on peer outputs)
                stage1_corr_loss, _, _ = calc_categories_loss(
                    y1_corr_probs,
                    y1_true,
                    self.category_sizes.y1,
                    self.stage1_criterion,
                )

                # STAGE 2
                stage2_loss = self.stage2_criterion(y2_preds, y2_true)

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

                loss.backward()
                self.model.optimizer.step()

                self.calculate_metrics(
                    y1_probs_by_field, y1_true_by_field, y2_preds, y2_true
                )

            if epoch % SAVE_FREQUENCY == 0:
                self.evaluate()
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
            k = list(self.category_sizes.y1.keys())[i]

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

    def evaluate(self):
        """
        Output evaluation metrics
        """
        try:
            for k in self.category_sizes.y1.keys():
                for metric in self.stage1_metrics.values():
                    logger.info("Stage1 %s: %s", k, metric[k].compute())
                    metric[k].reset()

            for metric, _ in self.stage2_metrics.values():
                logger.info("Stage2: %s", metric.compute())
                metric.reset()

        except Exception as e:
            logger.warning("Failed to evaluate: %s", e)

    @staticmethod
    def train_from_trials(batch_size: int = BATCH_SIZE):
        trials = preprocess_inputs(
            fetch_trials("COMPLETED", limit=2000), QUANTITATIVE_TO_CATEGORY_FIELDS
        )
        input_dict, category_sizes = prepare_inputs(
            trials,
            batch_size,
            SINGLE_SELECT_CATEGORICAL_FIELDS,
            MULTI_SELECT_CATEGORICAL_FIELDS,
            TEXT_FIELDS,
            QUANTITATIVE_FIELDS,
            Y1_CATEGORICAL_FIELDS,
            Y2_FIELD,
        )

        model = ModelTrainer(input_dict, category_sizes)

        num_batches = round(len(trials) / batch_size)
        model.train(input_dict, num_batches)


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
