import logging
import os
import sys
from typing import Optional
import torch
import torch.nn as nn
from ignite.metrics import Precision, Recall


from .constants import (
    BATCH_SIZE,
    CATEGORICAL_FIELDS,
    CHECKPOINT_PATH,
    CATEGORICAL_FIELDS,
    LR,
    OPTIMIZER_CLASS,
    SAVE_FREQUENCY,
    TEXT_FIELDS,
    TRUE_THRESHOLD,
)
from .core import DNN
from .types import AllInput, DnnInput
from .utils import prepare_inputs


class ModelTrainer:
    """
    Trainable model
    """

    def __init__(
        self,
        dnn_input_dim: int,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Initialize model

        Args:
            dnn_input_dim (int): Input dimension for DNN
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to use
        """
        torch.device("mps")

        logging.info("DNN input dim: %s", dnn_input_dim)
        self.model = DNN(dnn_input_dim, round(dnn_input_dim / 2))
        self.optimizer = optimizer or OPTIMIZER_CLASS(self.model.parameters(), lr=LR)
        self.criterion = nn.BCEWithLogitsLoss()
        self.dnn_input_dim = dnn_input_dim

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
                batch: AllInput = {k: v[i] for k, v in input_dict.items()}  # type: ignore
                pred = self.model(batch["x1"])
                loss = self.criterion(pred, batch["y"])  # max 0.497 min 0.162

                logging.info("Prediction loss: %s", loss)

                # backprop
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()

                # update status
                y_pred = pred > TRUE_THRESHOLD
                y_true = batch["y"] > TRUE_THRESHOLD
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
        trials = []
        input_dict = prepare_inputs(trials, BATCH_SIZE, CATEGORICAL_FIELDS, TEXT_FIELDS)
        dnn_input_dim = input_dict["x1"].size(2)
        model = ModelTrainer(dnn_input_dim)
        model.train(input_dict)


def main():
    ModelTrainer.train_from_trials()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m core.models.clindev.trainer
            """
        )
        sys.exit()

    main()
