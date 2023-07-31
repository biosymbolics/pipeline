"""
Patent Probability of Success (PoS) model(s)
"""
import logging
import os
import random
import sys
from typing import Any, Optional, Sequence, cast
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

from clients.patents import patent_client
from common.utils.tensor import pad_or_truncate_to_size
from core.models.patent_pos.types import AllInput
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
)
from .utils import prepare_inputs


class DNN(nn.Module):
    """
    Contrastive DNN for patent classification
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dnn(x)

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return super().__call__(*args, **kwds)


class GNN(nn.Module):
    """
    Graph neural network for patent classification

    Where be the loss?
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        layers = nn.Sequential(self.conv1, nn.ReLU(), self.conv2)
        x = layers(x, edge_index)
        return x.mean(dim=0)

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return super().__call__(*args, **kwds)


# Combine modules
class CombinedModel(nn.Module):
    """
    Combined model (DNN + GNN) for patent classification

    Loading:
        >>> import torch; import system; system.initialize();
        >>> from core.models.patent_pos.core import CombinedModel
        >>> model = CombinedModel(4140, 480)
        >>> checkpoint = torch.load("patent_model_checkpoints/checkpoint_15.pt", map_location=torch.device('mps'))
        >>> model.load_state_dict(checkpoint["model_state_dict"])
        >>> model.eval()
        >>> torch.save(model, 'patent-model.pt')
    """

    def __init__(
        self,
        dnn_input_dim: int,
        gnn_input_dim: int,
        dnn_hidden_dim: int = 64,
        gnn_hidden_dim: int = 64,
    ):
        torch.device("mps")
        super().__init__()
        combo_hidden_dim = dnn_hidden_dim  # + gnn_hidden_dim
        midway_hidden_dim = round(combo_hidden_dim / 2)
        self.dnn = DNN(dnn_input_dim, dnn_hidden_dim)
        self.gnn = GNN(gnn_input_dim, gnn_hidden_dim)
        self.fc1 = nn.Linear(combo_hidden_dim, midway_hidden_dim)
        self.fc2 = nn.Linear(midway_hidden_dim, 1)

    def forward(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        edge_index: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns:
            torch.Tensor (torch.Size([BATCH_SIZE]))
        """
        is_nan = any(x1.isnan().flatten())

        if is_nan:
            raise ValueError("X1 is nan: ", x1[0:5])

        # gnn_emb = self.gnn(x2, edge_index).unsqueeze(0).repeat(BATCH_SIZE, 1)
        # x = torch.cat([dnn_emb, gnn_emb], dim=1)

        layers = nn.Sequential(self.dnn, self.fc1, self.fc2, nn.Sigmoid())

        return layers(x1).squeeze()


class ModelTrainer:
    """
    Trainable combined model for patent classification
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
            input_dict (AllInput): Dictionary of input tensors
            model (Optional[CombinedModel]): Model to train
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to use
        """
        torch.device("mps")

        logging.info("DNN input dim: %s", dnn_input_dim)
        self.model = model or CombinedModel(dnn_input_dim, gnn_input_dim)
        self.optimizer = optimizer or OPTIMIZER_CLASS(self.model.parameters(), lr=LR)
        self.criterion = nn.BCEWithLogitsLoss()

        self.dnn_input_dim = dnn_input_dim

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
            # 'loss': LOSS,
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
        num_epochs: int = 20,
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
                batch = {k: v[i] for k, v in input_dict.items()}  # type: ignore
                pred = self.model(batch["x1"], batch["x2"], batch["edge_index"])
                loss = self.criterion(pred, batch["y"])  # contrastive??
                logging.info(
                    "Prediction size: %s, loss %s",
                    pred.size(),
                    loss,
                )
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                self.optimizer.step()
            if epoch % SAVE_FREQUENCY == 0:
                self.save_checkpoint(epoch)


class ModelPredictor:
    """
    Class for model prediction

    Example:
    ```
    from core.models.patent_pos import ModelPredictor; from clients.patents import patent_client
    patents = patent_client.search(["asthma"], True, 0, "medium", max_results=1000)
    predictor = ModelPredictor()
    preds = predictor(patents)
    ```
    """

    def __init__(
        self,
        checkpoint: str = "checkpoint_15.pt",
        dnn_input_dim: int = 5040,
        gnn_input_dim: int = 480,
    ):
        self.dnn_input_dim = dnn_input_dim
        self.gnn_input_dim = gnn_input_dim
        model = CombinedModel(dnn_input_dim, gnn_input_dim)
        checkpoint_obj = torch.load(
            f"{CHECKPOINT_PATH}/{checkpoint}",
            map_location=torch.device("mps"),
        )
        model.load_state_dict(checkpoint_obj["model_state_dict"])
        model.eval()
        self.model = model

    def __call__(self, *args, **kwargs):
        return self.predict(*args, **kwargs)

    def predict_tensor(self, input_dict: AllInput) -> list[float]:
        """
        Predict probability of success for a given tensor input

        Args:
            input_dict (AllInput): Dictionary of input tensors
        """
        x1_padded = pad_or_truncate_to_size(
            input_dict["x1"], (input_dict["x1"].size(0), BATCH_SIZE, self.dnn_input_dim)
        )
        num_batches = x1_padded.size(0)

        output = torch.flatten(
            torch.cat([self.model(x1_padded[i]) for i in range(num_batches)])
        )

        return [output[i].item() for i in range(output.size(0))]

    def predict(self, patents: list[PatentApplication]) -> list[float]:
        """
        Predict probability of success for a given input

        Args:
            patents (list[PatentApplication]): List of patent applications

        Returns:
            list[float]: Probabilities of success
        """
        input_dict = prepare_inputs(
            patents, BATCH_SIZE, CATEGORICAL_FIELDS, TEXT_FIELDS, GNN_CATEGORICAL_FIELDS
        )

        output = self.predict_tensor(input_dict)

        for i, patent in enumerate(patents):
            logging.info(
                "Patent %s (%s): %s (%s)",
                patent["publication_number"],
                (output[i] > 0.5),
                patent["title"],
                output[i],
            )

        return output


def main():
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


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m core.models.patent_pos.core \nTrains patent PoS (probability of success) model"
        )
        sys.exit()

    main()
