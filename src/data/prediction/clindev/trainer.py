from itertools import accumulate
import logging
import math
import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from ignite.metrics import Precision, Recall, MeanAbsoluteError

import system

system.initialize()

from clients.trials import fetch_trials


from .constants import (
    BATCH_SIZE,
    MULTI_SELECT_CATEGORICAL_FIELDS,
    SINGLE_SELECT_CATEGORICAL_FIELDS,
    CHECKPOINT_PATH,
    SAVE_FREQUENCY,
    TEXT_FIELDS,
    Y1_CATEGORICAL_FIELDS,
    Y2_FIELD,
)
from .model import TwoStageModel
from .types import DnnInput, TwoStageModelSizes
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
        y1_category_size_map: dict[str, int],
    ):
        """
        Initialize model

        Args:
            input_dim (int): Input dimension for DNN
            stage1_output_dim (int): Output dimension for stage 1 DNN
        """
        torch.device("mps")

        self.input_dim = 128  # arbitrary

        self.y1_category_size_map = y1_category_size_map
        self.y1_field_indexes = tuple(n for n in range(len(Y1_CATEGORICAL_FIELDS)))

        sizes = TwoStageModelSizes(
            multi_select_input=math.prod(input_dict["multi_select_x"].shape[2:]),
            single_select_input=math.prod(input_dict["single_select_x"].shape[2:]),
            text_input=math.prod(input_dict["text_x"].shape[2:])
            if input_dict["text_x"]
            else 0,
            stage1_input=self.input_dim,
            stage2_input=math.prod(input_dict["y1"].shape[2:]),
            stage1_hidden=round(self.input_dim / 2),
            stage1_embedded_output=math.prod(input_dict["y1"].shape[2:]),
            stage1_output_map=y1_category_size_map,
            stage2_hidden=round(self.input_dim / 2),
            stage2_output=1,
        )
        logging.info("Model sizes: %s", sizes)

        self.model = TwoStageModel(sizes)
        self.optimizer = self.model.optimizer
        self.stage1_criterion = nn.CrossEntropyLoss()
        self.stage2_criterion = nn.MSELoss()

        logger.info("Initialized model with input dim of %s", self.input_dim)

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
        num_batches = input_dict["multi_select_x"].size(0)

        for epoch in range(start_epoch, num_epochs):
            logging.info("Starting epoch %s", epoch)
            for i in range(num_batches):
                logging.info("Starting batch %s out of %s", i, num_batches)
                batch: DnnInput = {k: v[i] for k, v in input_dict.items() if v is not None}  # type: ignore
                batch_size = batch["multi_select_x"].size(0)

                y1_true = batch["y1"]
                y2_true = batch["y2"]

                # y1_logits = torch.Size([256, 64])
                y1_logits, y2_logits, y1_probs, y1_probs_dict = self.model(
                    batch["multi_select_x"].view(batch_size, -1),
                    batch["single_select_x"].view(batch_size, -1),
                    batch["text_x"].view(batch_size, -1)
                    if "text_x" in batch and batch["text_x"] is not None
                    else None,
                )
                y1_preds = y1_logits.view(y1_true.shape)

                self.optimizer.zero_grad()

                # STAGE 1
                # indexes with which to split y1_probs into 1 tensor per field
                y1_idxs = list(
                    accumulate(
                        list(self.y1_category_size_map.values()), lambda x, y: x + y
                    )
                )[:-1]

                y1_probs_by_field = torch.tensor_split(y1_probs, y1_idxs, dim=1)
                y1_true_by_field = torch.split(batch["y1_categories"], 1, dim=1)

                field_losses = [
                    self.stage1_criterion(
                        y1_by_field.float(),
                        y1_true_set.squeeze(),
                    )
                    for y1_by_field, y1_true_set in zip(
                        y1_probs_by_field, y1_true_by_field
                    )
                ]

                stage1_loss = torch.stack(field_losses).sum()
                print("stage1_loss grad fn", stage1_loss.grad_fn)

                logging.info("Stage 1 loss: %s", stage1_loss)

                # STAGE 2
                stage2_loss = self.stage2_criterion(y2_logits, y2_true)
                logging.info("Stage 2 loss: %s", stage2_loss)

                # combine loss and backprop both at the same time
                loss = stage1_loss + (math.log(stage2_loss) * 0.01)
                logging.info("Total loss: %s", loss)

                loss.backward(retain_graph=True)
                self.optimizer.step()

                self.calculate_metrics(batch, y1_preds, y1_probs, y2_logits)

            if epoch % SAVE_FREQUENCY == 0:
                self.evaluate()
                self.save_checkpoint(epoch)

    def calculate_metrics(
        self,
        batch,
        y1_preds: torch.Tensor,
        y1_probs: torch.Tensor,
        y2_preds: torch.Tensor,
    ):
        """
        Calculate metrics for a batch
        """
        y1_true = batch["y1"]
        y2_true = batch["y2"]
        print(y1_preds.shape, y1_true.shape, y2_preds.shape, y2_true.shape)
        self.stage1_mae.update((y1_preds, y1_true))
        self.stage2_mae.update((y2_preds, y2_true))
        self.calculate_discrete_metrics(batch, y1_preds, y1_probs)

    def calculate_discrete_metrics(
        self, batch: DnnInput, y1_pred: torch.Tensor, y1_probs: torch.Tensor
    ):
        """
        Calculate discrete metrics for a batch (precision/recall/f1)
        """
        print("YPROBS", y1_probs.shape, y1_probs[0:2])
        y1_pred_cats = (y1_probs > 0.5).float()
        y1_true_cats = batch["y1_categories"]

        y1_preds_oh = (
            y1_pred_cats  # F.one_hot(y1_pred_cats.squeeze(), num_classes=num_classes)
        )
        y1_true_oh = (
            y1_true_cats  # F.one_hot(y1_true_cats.squeeze(), num_classes=num_classes)
        )

        # y1_true_oh: torch.Size([256, 3, 5]), y1_preds_oh: torch.Size([256, 5])
        logger.info(
            "y1_true_oh: %s (%s), y1_preds_oh: %s (%s)",
            y1_true_oh.shape,
            y1_true_oh[0:2],
            y1_preds_oh.shape,
            y1_preds_oh[0:2],
        )
        self.stage1_precision.update((y1_true_oh, y1_preds_oh))
        self.stage1_recall.update((y1_true_oh, y1_preds_oh))

    def evaluate(self):
        """
        Output evaluation metrics (precision, recall, F1)
        """
        try:
            logging.info("Stage 1 Precision: %s", self.stage1_precision.compute())
            logging.info("Stage 1 Recall: %s", self.stage1_recall.compute())
            logging.info("Stage 1 MAE: %s", self.stage1_mae.compute())

            self.stage1_precision.reset()
            self.stage1_recall.reset()
            self.stage1_mae.reset()

            logging.info("Stage 2 MAE: %s", self.stage2_mae.compute())
            self.stage2_mae.reset()

        except Exception as e:
            logging.warning("Failed to evaluate: %s", e)

    @staticmethod
    def train_from_trials():
        trials = fetch_trials("COMPLETED", limit=2000)
        input_dict, y1_category_size_map = prepare_inputs(
            trials,
            BATCH_SIZE,
            SINGLE_SELECT_CATEGORICAL_FIELDS,
            MULTI_SELECT_CATEGORICAL_FIELDS,
            TEXT_FIELDS,
            Y1_CATEGORICAL_FIELDS,
            Y2_FIELD,
        )

        model = ModelTrainer(input_dict, y1_category_size_map)
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
