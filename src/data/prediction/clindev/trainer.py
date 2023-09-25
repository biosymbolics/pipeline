from itertools import accumulate
import logging
import math
import os
import random
import sys
from typing import Sequence, cast
import torch
import torch.nn as nn
from ignite.metrics import (
    ClassificationReport,
    Accuracy,
    MeanAbsoluteError,
)

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
    SAVE_FREQUENCY,
    TEXT_FIELDS,
    Y1_CATEGORICAL_FIELDS,
    Y2_FIELD,
)
from .model import TwoStageModel
from .types import AllCategorySizes, DnnInput, TwoStageModelSizes
from .utils import prepare_inputs


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
            stage2_output=math.prod(input_dict["y2"].shape[2:]),
        )
        logging.info("Model sizes: %s", sizes)

        self.model = TwoStageModel(sizes)
        self.stage1_criterion = nn.CrossEntropyLoss()
        self.stage2_criterion = nn.MSELoss()

        self.stage1_cp = {
            k: ClassificationReport(output_dict=True) for k in category_sizes.y1.keys()
        }
        self.stage1_accuracy = {k: Accuracy() for k in category_sizes.y1.keys()}
        self.stage2_mae = MeanAbsoluteError()

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
            logging.info("Saved checkpoint %s", checkpoint_name)
        except Exception as e:
            logging.error("Failed to save checkpoint %s: %s", checkpoint_name, e)
            raise e

    def train(
        self,
        input_dict: DnnInput,
        num_batches: int,
        start_epoch: int = 0,
        num_epochs: int = 500,
    ):
        """
        Train model

        Args:
            input_dict (DnnInput): Dictionary of input tensors
            start_epoch (int, optional): Epoch to start training from. Defaults to 0.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 20.
        """
        for epoch in range(start_epoch, num_epochs):
            logging.info("Starting epoch %s", epoch)
            _input_dict = {f: v.detach().clone() for f, v in input_dict.items()}  # type: ignore
            for i in range(num_batches):
                logging.debug("Starting batch %s out of %s", i, num_batches)
                batch: DnnInput = cast(
                    DnnInput,
                    {f: v[i] for f, v in _input_dict.items() if v is not None},
                )

                # place before any loss calculation
                self.model.optimizer.zero_grad()

                y2_true = batch["y2"]

                _, y2_logits, y1_probs = self.model(
                    torch.split(batch["multi_select_x"], 1, dim=1),
                    torch.split(batch["single_select_x"], 1, dim=1),
                    batch["text_x"],
                    batch["quantitative_x"],
                )

                # STAGE 1
                # indexes with which to split y1_probs into 1 tensor per field
                y1_idxs = list(
                    accumulate(
                        list(self.category_sizes.y1.values()),
                        lambda x, y: x + y,
                    )
                )[:-1]

                y1_probs_by_field = torch.tensor_split(y1_probs, y1_idxs, dim=1)
                y1_true_by_field = [
                    y1.squeeze() for y1 in torch.split(batch["y1_categories"], 1, dim=1)
                ]

                stage1_loss = torch.stack(
                    [
                        self.stage1_criterion(y1_by_field.float(), y1_true_set)
                        for y1_by_field, y1_true_set in zip(
                            y1_probs_by_field,
                            y1_true_by_field,
                        )
                    ]
                ).sum()

                logging.debug(
                    "Batch %s Stage 1 loss: %s", i, stage1_loss.detach().item()
                )

                # STAGE 2
                # note: can be very large thus the log/0.5 when combining with stage 1
                stage2_loss = self.stage2_criterion(y2_logits, y2_true)
                logging.debug(
                    "Batch %s Stage 2 loss: %s", i, stage2_loss.detach().item()
                )

                # Total
                loss = stage1_loss + torch.mul(torch.log(stage2_loss), 0.5)
                logging.info("Batch %s Total loss: %s", i, loss.detach().item())

                loss.backward()
                self.model.optimizer.step()

                self.calculate_continuous_metrics(batch, y2_logits)
                self.calculate_discrete_metrics(y1_probs_by_field, y1_true_by_field)

            if epoch % SAVE_FREQUENCY == 0:
                self.evaluate()
                self.save_checkpoint(epoch)

    def calculate_continuous_metrics(
        self,
        batch,
        y2_preds: torch.Tensor,
    ):
        """
        Calculate metrics for a batch
        """
        y2_true = batch["y2"]
        self.stage2_mae.update((y2_preds, y2_true))

    def calculate_discrete_metrics(
        self,
        y1_logits_by_field: Sequence[torch.Tensor],
        y1_true_by_field: Sequence[torch.Tensor],
    ):
        """
        Calculate discrete metrics for a batch
        """

        for i, (y1_preds, y1_true) in enumerate(
            zip(y1_logits_by_field, y1_true_by_field)
        ):
            k = list(self.category_sizes.y1.keys())[i]
            # ignite.metrics uses 64 bit, not supported by MPS
            y1_pred_cats = y1_preds.detach().to("cpu")

            self.stage1_cp[k].update((y1_pred_cats, y1_true.to("cpu")))
            self.stage1_accuracy[k].update((y1_pred_cats, y1_true.to("cpu")))

    def evaluate(self):
        """
        Output evaluation metrics
        """
        try:
            for k in self.category_sizes.y1.keys():
                logging.info("Stage1 %s Metrics: %s", k, self.stage1_cp[k].compute())
                logging.info(
                    "Stage1 %s Accuracy: %s", k, self.stage1_accuracy[k].compute()
                )
                self.stage1_cp[k].reset()
                self.stage1_accuracy[k].reset()

            logging.info("Stage2 MAE: %s", self.stage2_mae.compute())
            self.stage2_mae.reset()

        except Exception as e:
            logging.warning("Failed to evaluate: %s", e)

    @staticmethod
    def train_from_trials(batch_size: int = BATCH_SIZE):
        trials = sorted(
            fetch_trials("COMPLETED", limit=20000), key=lambda x: random.random()
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
