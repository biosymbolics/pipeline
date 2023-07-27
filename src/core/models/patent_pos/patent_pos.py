"""
Patent Probability of Success (PoS) model(s)
"""
import logging
import os
import sys
from typing import Any, Optional, Sequence, cast
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

from common.utils.list import batch, batch_as_tensors
from clients.patents import patent_client
from typings.patents import ApprovedPatentApplication as PatentApplication
from typings.core import Primitive

from .constants import (
    BATCH_SIZE,
    CHECKPOINT_PATH,
    EMBEDDING_DIM,
    LR,
    OPTIMIZER_CLASS,
    SAVE_FREQUENCY,
    CATEGORICAL_FIELDS,
    TEXT_FIELDS,
    GNN_CATEGORICAL_FIELDS,
)
from .types import AllInput, DnnInput, GnnInput
from .utils import get_feature_embeddings, get_input_dim, is_tensor_list

# Query for approval data
# product p, active_ingredient ai, synonyms syns, approval a
"""
select
    p.ndc_product_code as ndc,
    (array_agg(distinct p.generic_name))[1] as generic_name,
    (array_agg(distinct p.product_name))[1] as brand_name,
    (array_agg(distinct p.marketing_status))[1] as status,
    (array_agg(distinct active_ingredient_count))[1] as active_ingredient_count,
    (array_agg(distinct route))[1] as route,
    (array_agg(distinct s.name)) as substance_names,
    (array_agg(distinct a.type)) as approval_types,
    (array_agg(distinct a.approval)) as approval_dates,
    (array_agg(distinct a.applicant)) as applicants
from structures s
LEFT JOIN approval a on a.struct_id=s.id
LEFT JOIN active_ingredient ai on ai.struct_id=s.id
LEFT JOIN product p on p.ndc_product_code=ai.ndc_product_code
LEFT JOIN synonyms syns on syns.id=s.id
where (syns.name ilike '%elexacaftor%' or p.generic_name ilike '%elexacaftor%' or p.product_name ilike '%elexacaftor%')
group by p.ndc_product_code;
"""


class DNN(nn.Module):
    """
    Contrastive DNN for patent classification
    """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x) -> torch.Tensor:
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

    def forward(self, x, edge_index) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x.mean(dim=0)

    def __call__(self, *args: Any, **kwds: Any) -> torch.Tensor:
        return super().__call__(*args, **kwds)


# Combine modules
class CombinedModel(nn.Module):
    """
    Combined model (DNN + GNN) for patent classification
    """

    def __init__(
        self,
        dnn_input_dim: int,
        gnn_input_dim: int,
        dnn_hidden_dim: int = 64,
        gnn_hidden_dim: int = 64,
    ):
        super().__init__()
        combo_hidden_dim = dnn_hidden_dim  # + gnn_hidden_dim
        midway_hidden_dim = round(combo_hidden_dim / 2)
        self.dnn = DNN(dnn_input_dim, dnn_hidden_dim)
        self.gnn = GNN(gnn_input_dim, gnn_hidden_dim)
        self.fc1 = nn.Linear(combo_hidden_dim, midway_hidden_dim)
        self.fc2 = nn.Linear(midway_hidden_dim, 1)

    def forward(self, x1, x2, edge_index) -> torch.Tensor:
        """
        Returns:
            torch.Tensor (torch.Size([BATCH_SIZE]))
        """
        dnn_emb = self.dnn(x1)
        gnn_emb = self.gnn(x2, edge_index).unsqueeze(0).repeat(BATCH_SIZE, 1)

        logging.info("DNN (%s), GNN (%s)", dnn_emb.shape, gnn_emb.shape)
        # x = torch.cat([dnn_emb, gnn_emb], dim=1)
        x = dnn_emb
        x = self.fc1(x)
        x = self.fc2(x)
        return x.squeeze()


class TrainableCombinedModel:
    """
    Trainable combined model for patent classification
    """

    def __init__(
        self,
        batch_size: int = BATCH_SIZE,
        model: Optional[CombinedModel] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        gnn_categorical_fields: list[str] = GNN_CATEGORICAL_FIELDS,
        dnn_categorical_fields: list[str] = CATEGORICAL_FIELDS,
        dnn_text_fields: list[str] = TEXT_FIELDS,
    ):
        """
        Initialize model

        Args:
            model (Optional[CombinedModel]): Model to train
            optimizer (Optional[torch.optim.Optimizer]): Optimizer to use
        """
        torch.device("mps")
        self.embedding_dim = EMBEDDING_DIM
        self.batch_size = batch_size

        dnn_input_dim = get_input_dim(
            dnn_categorical_fields, dnn_text_fields, self.embedding_dim
        )
        gnn_input_dim = get_input_dim(gnn_categorical_fields, [], self.embedding_dim)
        self.model = model or CombinedModel(dnn_input_dim, gnn_input_dim)
        self.optimizer = optimizer or OPTIMIZER_CLASS(self.model.parameters(), lr=LR)
        self.criterion = nn.BCEWithLogitsLoss()
        self.gnn_categorical_fields = gnn_categorical_fields
        self.dnn_categorical_fields = dnn_categorical_fields
        self.dnn_text_fields = dnn_text_fields

    def __call__(self, *args, **kwargs):
        """
        Alias for self.train
        """
        self.train(*args, **kwargs)

    def __batch(self, items: list[torch.Tensor] | list[Primitive]) -> torch.Tensor:

        if not is_tensor_list(items):
            # then list of primitives
            batches = batch_as_tensors(cast(list[Primitive], items), self.batch_size)
            num_dims = 1
        else:
            num_dims = len(items[0].size()) + 1
            batches = batch(items, self.batch_size)
            batches = [torch.stack(b) for b in batches]

        def get_batch_pad(batch: torch.Tensor):
            if num_dims == 1:
                return (0, self.batch_size - batch.size(0))

            if num_dims == 2:
                return (0, 0, 0, self.batch_size - batch.size(0))

            raise ValueError("Unsupported number of dimensions: %s" % num_dims)

        batches = [F.pad(b, get_batch_pad(b)) for b in batches]

        logging.info("Batches: %s (%s)", len(batches), [b.size() for b in batches])
        return torch.stack(batches)

    def __resize_and_batch(self, embeddings: list[torch.Tensor]):
        """
        Size embeddings into batches
        """
        max_len = max(e.size(0) for e in embeddings)
        padded_emb = [F.pad(f, (0, max_len - f.size(0))) for f in embeddings]
        return self.__batch(padded_emb)

    def __prepare_dnn_data(self, patents: Sequence[PatentApplication]) -> DnnInput:
        """
        Prepare data for DNN

        Args:
            patents (Sequence[PatentApplication]): List of patents

        Returns:
            DnnInput: data shaped for DNN input layer
        """
        embeddings = get_feature_embeddings(
            patents, self.dnn_categorical_fields, self.dnn_text_fields
        )
        x1 = self.__resize_and_batch(embeddings)
        logging.info("X1: %s", x1.size())

        y = self.__batch([patent["approval_date"] is not None for patent in patents])
        logging.info("Y: %s", y.size())  # Y: torch.Size([2000])
        return {"x1": x1, "y": y}

    def __prepare_gnn_input(
        self,
        patents: Sequence[PatentApplication],
    ) -> GnnInput:
        """
        Prepare inputs for GNN

        Args:
            patents (Sequence[PatentApplication]): List of patents

        Returns:
            GNNInput: data shaped for GNN input layer
        """
        # TODO: enirch with pathways, targets, disease pathways
        # GNN input features for this node
        embeddings = get_feature_embeddings(patents, self.gnn_categorical_fields)
        x2 = self.__resize_and_batch(embeddings)
        edge_index = torch.tensor([[i, i] for i in range(len(patents))])
        logging.info("X2: %s, EI", x2.size(), edge_index.size())

        return {"x2": x2, "edge_index": edge_index}

    def __prepare_inputs(self, patents: Sequence[PatentApplication]) -> AllInput:
        """
        Prepare inputs for model
        """
        return cast(
            AllInput,
            {**self.__prepare_dnn_data(patents), **self.__prepare_gnn_input(patents)},
        )

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

    def train(
        self,
        patents: Sequence[PatentApplication],
        start_epoch: int = 0,
        num_epochs: int = 20,
    ):
        """
        Train model

        Args:
            patents (Sequence[PatentApplication]): List of patents
            start_epoch (int, optional): Epoch to start training from. Defaults to 0.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 20.
        """
        input_dict = self.__prepare_inputs(patents)
        num_batches = input_dict["x1"].size(0)

        for epoch in range(start_epoch, num_epochs):
            logging.info("Starting epoch %s", epoch)
            for i in range(num_batches):
                batch = {k: v[i] for k, v in input_dict.items()}  # type: ignore
                logging.info("Starting batch %s out of %s", i, num_batches)
                self.optimizer.zero_grad()
                pred = self.model(batch["x1"], batch["x2"], batch["edge_index"])
                print(pred)
                print(batch["y"][0:5])
                print(
                    batch["x1"][0:5]
                )  # tensor([[ 0.4233, -0.9974, -0.6870,  ...,  0.0000,  0.0000,  0.0000]
                print(
                    batch["x2"][0:5]
                )  # tensor([[-2.0202,  0.3456, -0.3380,  ...,  0.0000,  0.0000,  0.0000],
                print(batch["edge_index"][0:5])  # tensor([0, 0])
                print("PRED:", pred.size(), batch["y"].size())
                loss = self.criterion(pred, batch["y"])  # contrastive??
                loss.backward()
                self.optimizer.step()
            if epoch % SAVE_FREQUENCY == 0:
                self.save_checkpoint(epoch)

    @classmethod
    def load_checkpoint(
        cls, checkpoint_name: str, patents: Optional[Sequence[PatentApplication]] = None
    ):
        """
        Load model from checkpoint. If patents provided, will start training from the next epoch

        Args:
            patents (Sequence[PatentApplication]): List of patents
            checkpoint_name (str): Checkpoint from which to resume
        """
        logging.info("Loading checkpoint %s", checkpoint_name)
        model = CombinedModel(100, 100)  # TODO!!
        checkpoint_file = os.path.join(CHECKPOINT_PATH, checkpoint_name)

        if not os.path.exists(checkpoint_file):
            raise Exception(f"Checkpoint {checkpoint_name} does not exist")

        checkpoint = torch.load(checkpoint_file)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer = OPTIMIZER_CLASS(model.parameters(), lr=LR)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        logging.info("Loaded checkpoint %s", checkpoint_name)

        trainable_model = TrainableCombinedModel(BATCH_SIZE, model, optimizer)

        if patents:
            trainable_model.train(patents, start_epoch=checkpoint["epoch"] + 1)

        return trainable_model


def main():
    patents = cast(
        Sequence[PatentApplication], patent_client.search(["asthma"], True, 0, "medium")
    )
    print(patents[0])
    model = TrainableCombinedModel()
    model.train(patents)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 -m core.models.patent_pos.patent_pos \nTrains patent PoS (probability of success) model"
        )
        sys.exit()

    main()
