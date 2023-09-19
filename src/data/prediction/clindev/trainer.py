import logging
import math
import os
import sys
import torch
import torch.nn as nn
from ignite.metrics import Precision, Recall, MeanAbsoluteError

import system

system.initialize()

from clients.trials import fetch_trials
from utils.tensor import reduce_last_dim, reverse_embedding


from .constants import (
    BATCH_SIZE,
    CATEGORICAL_FIELDS,
    CHECKPOINT_PATH,
    CATEGORICAL_FIELDS,
    SAVE_FREQUENCY,
    TEXT_FIELDS,
    Y1_CATEGORICAL_FIELDS,
    Y2_FIELD,
)
from .model import TwoStageModel
from .types import DnnInput
from .utils import prepare_inputs


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ModelTrainer:
    """
    Trainable model
    """

    def __init__(
        self,
        input_dim: int,
        stage1_output_dim: int,
        original_shape: tuple[int, ...],
        embedding_weights: list[torch.Tensor],
    ):
        """
        Initialize model

        Args:
            input_dim (int): Input dimension for DNN
            stage1_output_dim (int): Output dimension for stage 1 DNN
        """
        torch.device("mps")

        self.input_dim = input_dim
        self.y1_field_indexes = tuple(n for n in range(len(Y1_CATEGORICAL_FIELDS)))
        self.embedding_weights = embedding_weights
        self.y1_embedding_weights = [
            w
            for i, w in enumerate(self.embedding_weights)
            if i in self.y1_field_indexes
        ]

        self.model = TwoStageModel(
            self.y1_field_indexes,
            input_dim,
            original_shape=original_shape,
            stage1_output_size=stage1_output_dim,
        )

        self.optimizer = self.model.optimizer

        self.stage1_criterion = nn.MSELoss()  # nn.CrossEntropyLoss()
        self.stage2_criterion = nn.MSELoss()

        logger.info("Initialized model with input dim of %s", input_dim)

        self.stage1_precision = Precision(average=None, is_multilabel=True)
        self.stage1_recall = Recall(average=None, is_multilabel=True)
        self.stage1_mae = MeanAbsoluteError()
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
            "optimizer_state_dict": self.optimizer.state_dict(),
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
        num_batches = input_dict["x1"].size(0)

        for epoch in range(start_epoch, num_epochs):
            logging.info("Starting epoch %s", epoch)
            for i in range(num_batches):
                logging.info("Starting batch %s out of %s", i, num_batches)
                batch: DnnInput = {k: v[i] for k, v in input_dict.items()}  # type: ignore

                input = batch["x1"].view(batch["x1"].size(0), -1)
                y1_true = batch["y1"]
                y2_true = batch["y2"]

                y1_logits, y2_logits = self.model(input)
                y1_preds = y1_logits.view(y1_true.shape)

                self.optimizer.zero_grad()

                # STAGE 1
                stage1_loss = self.stage1_criterion(y1_preds, y1_true)
                logging.info("Stage 1 loss: %s", stage1_loss)

                # STAGE 2
                stage2_loss = self.stage2_criterion(y2_logits, y2_true)
                logging.info("Stage 2 loss: %s", stage2_loss)

                # combine loss and backprop both at the same time
                loss = stage1_loss + stage2_loss
                loss.backward(retain_graph=True)
                self.optimizer.step()

                self.calculate_metrics(batch, y1_preds, y2_logits)

            if epoch % SAVE_FREQUENCY == 0:
                self.evaluate()
                self.save_checkpoint(epoch)

    def calculate_metrics(self, batch, y1_preds: torch.Tensor, y2_preds: torch.Tensor):
        y1_true = batch["y1"]
        y2_true = batch["y2"]
        self.stage1_mae.update((y1_preds, y1_true))
        self.stage2_mae.update((y2_preds, y2_true))
        self.calculate_discrete_metrics(batch, y1_preds)

    def calculate_discrete_metrics(self, batch, y1_pred: torch.Tensor):
        y1_pred_cats = reverse_embedding(y1_pred, self.y1_embedding_weights)
        y1_preds_oh = reduce_last_dim(y1_pred_cats, force=True, return_one_hot=True)

        y1_true_cats = batch["y1_categories"]
        y1_true_oh = reduce_last_dim(y1_true_cats, return_one_hot=True)

        logger.info("y1_true_oh: %s, y1_preds_oh", y1_true_oh.shape, y1_preds_oh.shape)
        self.stage1_precision.update((y1_true_oh, y1_preds_oh))
        self.stage1_recall.update((y1_true_oh, y1_preds_oh))

    def evaluate(self):
        """
        Output evaluation metrics (precision, recall, F1)
        """
        try:
            print("Disabled evaluation")
            # logging.info("Stage 1 Precision: %s", self.stage1_precision.compute())
            # logging.info("Stage 1 Recall: %s", self.stage1_recall.compute())
            logging.info("Stage 1 MAE: %s", self.stage1_mae.compute())

            # self.stage1_precision.reset()
            # self.stage1_recall.reset()
            self.stage1_mae.reset()

            logging.info("Stage 2 MAE: %s", self.stage2_mae.compute())
            self.stage2_mae.reset()

        except Exception as e:
            logging.warning("Failed to evaluate: %s", e)

    @staticmethod
    def train_from_trials():
        trials = fetch_trials("COMPLETED", limit=2000)  # 96001
        input_dict, embedding_weights = prepare_inputs(
            trials,
            BATCH_SIZE,
            CATEGORICAL_FIELDS,
            TEXT_FIELDS,
            Y1_CATEGORICAL_FIELDS,
            Y2_FIELD,
        )
        input_dim = math.prod(input_dict["x1"].shape[2:])
        # size of y1, which is derived from x1 (thus includes embeds)
        stage1_output_dim = math.prod(input_dict["y1"].shape[2:])
        model = ModelTrainer(
            input_dim, stage1_output_dim, input_dict["x1"].shape[1:], embedding_weights
        )
        model.train(input_dict)


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
