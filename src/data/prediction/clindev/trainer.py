import logging
import math
import os
import sys
import torch
import torch.nn as nn
from ignite.metrics import Precision, Recall, MeanAbsoluteError
from data.prediction.constants import DEFAULT_BATCH_SIZE

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
        target_idxs: tuple[int, ...],
        input_dim: int,
        stage1_output_dim: int,
        original_shape: tuple[int, ...],
    ):
        """
        Initialize model

        Args:
            input_dim (int): Input dimension for DNN
            stage1_output_dim (int): Output dimension for stage 1 DNN
        """
        torch.device("mps")

        self.input_dim = input_dim
        self.model = TwoStageModel(
            target_idxs,
            input_dim,
            original_shape=original_shape,
            stage1_output_size=stage1_output_dim,
        )

        self.optimizer = self.model.optimizer

        self.stage1_criterion = nn.MSELoss()  # nn.CrossEntropyLoss()
        # self.stage2_criterion = nn.MSELoss()

        logger.info("Initialized model with input dim of %s", input_dim)

        self.stage1_precision = Precision(is_multilabel=True)
        self.stage1_recall = Recall(is_multilabel=True)
        self.stage1_mae = MeanAbsoluteError()
        # self.stage2_mae = MeanAbsoluteError()

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
        embedding_weights,
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
                y1_logits, y2_logits = self.model(input)

                y1_pred_embeds = y1_logits.view(-1, 3, 48)  # 3, 6, 8
                y1_preds = y1_pred_embeds

                y1_true = batch["y1"]  # with embeds

                # y1_true = batch["y1_categories"] # just encoded
                # Project logits into embedding space
                # y1_preds = y1_pred_embeds @ embedding_weights.transpose(0, 1)
                # hypothetically those can now be compared with CEL

                # print(
                #     "y1_preds",
                #     y1_preds.size(),
                #     y1_preds[0:1],
                #     "Y1true",
                #     y1_true.size(),
                #     y1_true[0:1],
                # )

                self.optimizer.zero_grad()

                # STAGE 1
                stage1_loss = self.stage1_criterion(y1_preds, y1_true)
                logging.info("Stage 1 loss: %s", stage1_loss)

                # STAGE 2
                # stage2_loss = self.stage2_criterion(y2_logits, batch["y2"])
                # logging.info("Stage 2 loss: %s", stage2_loss)

                # combine loss and backprop both at the same time
                loss = stage1_loss.requires_grad_()  # + stage2_loss
                loss.backward(retain_graph=True)
                self.optimizer.step()

                self.stage1_mae.update((y1_preds, batch["y1"]))
                # self.stage1_precision.update((y1_preds, y1_true))
                # self.stage1_recall.update((y1_preds, y1_true))
                # self.stage2_mae.update((y2_logits, batch["y2"]))

                self.evaluate()
            if epoch % SAVE_FREQUENCY == 0:
                self.evaluate()
                self.save_checkpoint(epoch)

    def evaluate(self):
        """
        Output evaluation metrics (precision, recall, F1)
        """
        try:
            # logging.info("Stage 1 Precision: %s", self.stage1_precision.compute())
            # logging.info("Stage 1 Recall: %s", self.stage1_recall.compute())

            # self.stage1_precision.reset()
            # self.stage1_recall.reset()
            logging.info("Stage 1 MAE: %s", self.stage1_mae.compute())
            self.stage1_mae.reset()

            # logging.info("Stage 2 MAE: %s", self.stage2_mae.compute())
            # self.stage2_mae.reset()

        except Exception as e:
            logging.warning("Failed to evaluate: %s", e)

    @staticmethod
    def train_from_trials():
        trials = fetch_trials("COMPLETED", limit=20000)  # 96001
        input_dict, embedding_layer = prepare_inputs(
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
        y1_field_indexes = tuple(n for n in range(len(Y1_CATEGORICAL_FIELDS)))
        model = ModelTrainer(
            y1_field_indexes, input_dim, stage1_output_dim, input_dict["x1"].shape[1:]
        )
        model.train(input_dict, embedding_layer)


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
