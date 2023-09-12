import logging
import os
import sys
import torch
import torch.nn as nn
from ignite.metrics import Precision, Recall

import system

system.initialize()

from clients.trials import fetch_trials


from .constants import (
    BATCH_SIZE,
    CATEGORICAL_FIELDS,
    CHECKPOINT_PATH,
    CATEGORICAL_FIELDS,
    SAVE_FREQUENCY,
    TEXT_FIELDS,
    TRUE_THRESHOLD,
    Y1_CATEGORICAL_FIELDS,
    Y2_FIELD,
)
from .core import TwoStageModel
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
    ):
        """
        Initialize model

        Args:
            input_dim (int): Input dimension for DNN
            stage1_output_dim (int): Output dimension for stage 1 DNN
        """
        torch.device("mps")

        self.input_dim = input_dim
        self.model = TwoStageModel(input_dim, stage1_output_size=stage1_output_dim)

        self.stage1_optimizer = self.model.stage1_optimizer
        self.stage2_optimizer = self.model.stage2_optimizer

        self.stage1_criterion = nn.BCEWithLogitsLoss()
        self.stage2_criterion = nn.MSELoss()

        logger.info("Initialized model with input dim of %s", input_dim)

        self.precision = Precision(average=True)
        self.recall = Recall(average=True)
        self.f1 = 2 * (self.precision * self.recall) / (self.precision + self.recall)

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
            "stage1_optimizer_state_dict": self.stage1_optimizer.state_dict(),
            "stage2_optimizer_state_dict": self.stage2_optimizer.state_dict(),
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
                x1_pred, x2_pred = self.model(batch["x1"])

                # STAGE 1 backprop
                self.stage1_optimizer.zero_grad()
                stage1_loss = self.stage1_criterion(x1_pred, batch["y1"])
                logging.info("Stage 1 loss: %s", stage1_loss)
                stage1_loss.backward(retain_graph=True)
                self.stage1_optimizer.step()

                # STAGE 2 backprop
                self.stage2_optimizer.zero_grad()
                stage2_loss = self.stage2_criterion(x2_pred, batch["y2"])
                logging.info("Stage 2 loss: %s", stage2_loss)
                stage2_loss.backward(retain_graph=True)
                self.stage2_optimizer.step()

                # update status
                y_pred = x2_pred > TRUE_THRESHOLD
                y_true = batch["y2"] > TRUE_THRESHOLD
                self.precision.update((y_pred, y_true))
                self.recall.update((y_pred, y_true))
            if epoch % SAVE_FREQUENCY == 0:
                self.evaluate()
                self.save_checkpoint(epoch)

    def evaluate(self):
        """
        Output evaluation metrics (precision, recall, F1)
        """
        logging.info("Precision: %s", self.precision.compute())
        logging.info("Recall: %s", self.recall.compute())
        logging.info("F1 score: %s", self.f1.compute())

        self.precision.reset()
        self.recall.reset()

    @staticmethod
    def train_from_trials():
        trials = fetch_trials("COMPLETED", limit=100000)  # 96001
        input_dict = prepare_inputs(
            trials,
            BATCH_SIZE,
            CATEGORICAL_FIELDS,
            TEXT_FIELDS,
            Y1_CATEGORICAL_FIELDS,
            Y2_FIELD,
        )
        input_dim = input_dict["x1"].size(2)
        stage1_output_dim = input_dict["y1"].size(2)
        model = ModelTrainer(input_dim, stage1_output_dim)
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
