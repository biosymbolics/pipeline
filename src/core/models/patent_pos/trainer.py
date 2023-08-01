import logging
import os
import sys
from typing import Any, Optional, Sequence, cast
import torch
from ignite.metrics import Precision, Recall

from clients.patents import patent_client
from typings.patents import ApprovedPatentApplication as PatentApplication

from .constants import (
    BATCH_SIZE,
    CATEGORICAL_FIELDS,
    CHECKPOINT_PATH,
    GNN_CATEGORICAL_FIELDS,
    LR,
    OPTIMIZER_CLASS,
    SAVE_FREQUENCY,
    TEXT_FIELDS,
    TRUE_THRESHOLD,
)
from .core import CombinedModel
from .loss import FocalLoss
from .types import AllInput
from .utils import prepare_inputs


class ModelTrainer:
    """
    Trainable combined model for patent classification

    Usage:
        >>> ModelTrainer.train_from_patents()
    """

    def __init__(
        self,
        dnn_input_dim: int,
        gnn_input_dim: int,
        model: Optional[CombinedModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        """
        Initialize model

        Args:
            dnn_input_dim (int): Input dimension for DNN
            gnn_input_dim (int): Input dimension for GNN
            model (Optional[CombinedModel]): Model to train
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to use
        """
        torch.device("mps")

        logging.info("DNN input dim: %s", dnn_input_dim)
        self.model = model or CombinedModel(dnn_input_dim, gnn_input_dim)
        self.optimizer = optimizer or OPTIMIZER_CLASS(self.model.parameters(), lr=LR)
        # heavily penalize false negatives
        self.criterion = FocalLoss(alpha=0.75, gamma=20)  # nn.BCEWithLogitsLoss()
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
        input_dict: AllInput,
        start_epoch: int = 0,
        num_epochs: int = 100,
    ):
        """
        Train model

        Args:
            input_dict (AllInput): Dictionary of input tensors
            start_epoch (int, optional): Epoch to start training from. Defaults to 0.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 20.
        """
        num_batches = input_dict["x1"].size(0)

        for epoch in range(start_epoch, num_epochs):
            logging.info("Starting epoch %s", epoch)
            for i in range(num_batches):
                logging.info("Starting batch %s out of %s", i, num_batches)
                batch: AllInput = {k: v[i] for k, v in input_dict.items()}  # type: ignore
                pred = self.model(batch["x1"], batch["x2"], batch["edge_index"])
                loss = self.criterion(pred, batch["y"])  # max 0.497 min 0.162

                logging.info("Prediction loss (x100): %s", loss * 100)

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
    def train_from_patents():
        patents = cast(
            Sequence[PatentApplication],
            patent_client.search(["asthma"], True, 0, "medium", max_results=10000),
        )
        input_dict = prepare_inputs(
            patents, BATCH_SIZE, CATEGORICAL_FIELDS, TEXT_FIELDS, GNN_CATEGORICAL_FIELDS
        )
        dnn_input_dim = input_dict["x1"].size(2)
        gnn_input_dim = input_dict["x2"].size(2)
        model = ModelTrainer(dnn_input_dim, gnn_input_dim)
        model.train(input_dict)


def main():
    ModelTrainer.train_from_patents()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m core.models.patent_pos.trainer
            Trains patent PoS (probability of success) model
            """
        )
        sys.exit()

    main()
