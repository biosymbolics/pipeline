"""
Patent Probability of Success (PoS) model(s)
"""
from collections import namedtuple
import logging
import math
import os
import sys
from typing import Optional, Sequence, TypedDict, cast
from pydash import flatten
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
import polars as pl
from sklearn.preprocessing import LabelEncoder


from system import initialize

initialize()

from clients.spacy import Spacy
from common.utils.list import batch_dict
from clients.patents import patent_client
from typings.patents import ApprovedPatentApplication as PatentApplication

GnnInput = TypedDict("GnnInput", {"x2": torch.Tensor, "edge_index": torch.Tensor})
DnnInput = TypedDict("DnnInput", {"x1": torch.Tensor, "y": torch.Tensor})
AllInput = TypedDict(
    "AllInput",
    {
        "x1": torch.Tensor,
        "y": torch.Tensor,
        "x2": torch.Tensor,
        "edge_index": torch.Tensor,
    },
)

LR = 1e-3  # learning rate
CHECKPOINT_PATH = "patent_model_checkpoints"
OPTIMIZER_CLASS = torch.optim.Adam
SAVE_FREQUENCY = 2
EMBEDDING_DIM = 16  # good? should be small stuff

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

BATCH_SIZE = 256

CATEGORICAL_FIELDS = [
    "attributes",
    "compounds",
    "country",
    "diseases",
    "mechanisms",
    "ipc_codes",
]
TEXT_FIELDS = [
    "title",
    "abstract",
    # "claims",
    "assignees",
    "inventors",
]

GNN_CATEGORICAL_FIELDS = [
    "diseases",
    "compounds",
    "mechanisms",
]


class DNN(nn.Module):
    """
    Contrastive DNN for patent classification
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.dnn = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.dnn(x)


class GNN(nn.Module):
    """
    Graph neural network for patent classification

    Where be the loss?
    """

    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        return x.mean(dim=0)


# Combine modules
class CombinedModel(nn.Module):
    """
    Combined model (DNN + GNN) for patent classification
    """

    def __init__(
        self,
        dnn_input_dim: int,
        gnn_input_dim=64,
        dnn_hidden_dim=64,
        gnn_hidden_dim=64,
    ):
        super().__init__()
        combo_hidden_dim = gnn_hidden_dim + dnn_hidden_dim
        midway_hidden_dim = round(combo_hidden_dim / 2)
        self.dnn = DNN(dnn_input_dim, dnn_hidden_dim)
        self.gnn = GNN(gnn_input_dim, gnn_hidden_dim)
        self.fc1 = nn.Linear(combo_hidden_dim, midway_hidden_dim)
        self.fc2 = nn.Linear(midway_hidden_dim, 1)

    def forward(self, x1, x2, edge_index):
        dnn_emb = self.dnn(x1)
        gnn_emb = self.gnn(x2, edge_index)
        x = torch.cat([dnn_emb, gnn_emb], dim=1)
        x = self.fc1(x)
        x = self.fc2(x)
        return x


def get_input_dim(
    categorical_fields,
    text_fields=[],
    embedding_dim=EMBEDDING_DIM,
    text_embedding_dim=300,
):
    input_length = (len(categorical_fields) * embedding_dim) + (
        len(text_fields) * text_embedding_dim
    )
    return input_length


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
        self.model = model or CombinedModel(dnn_input_dim)
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

    def __get_feature_embeddings(
        self,
        patents: Sequence[PatentApplication],
        categorical_fields: list[str],
        text_fields: list[str] = [],
    ):
        """
        Get embeddings for patent features

        Args:
            patents (Sequence[PatentApplication]): List of patents
            categorical_fields (list[str]): List of fields to embed as categorical variables
            text_fields (list[str]): List of fields to embed as text
        """
        df = pl.from_dicts([patent for patent in patents])  # type: ignore
        size_map = dict(
            [
                (field, df.select(pl.col(field).flatten()).n_unique())
                for field in categorical_fields
            ]
        )
        embedding_layers = dict(
            [
                (field, nn.Embedding(size_map[field], self.embedding_dim))
                for field in categorical_fields
            ]
        )

        def encode_category(field):
            le = LabelEncoder()
            new_list = df.select(pl.col(field)).to_series().to_list()
            le.fit(flatten(new_list))
            values = [le.transform([x] if isinstance(x, str) else x) for x in new_list]
            return (le, values)

        # { 'mechanisms': [array([5]), array([5]) ...] }
        label_encoders = [
            (field, encode_category(field)) for field in categorical_fields
        ]
        # encoder_map = dict([(field, enc[0]) for field, enc in label_encoders])

        # 🐈
        CatTuple = namedtuple("CatTuple", categorical_fields)  # type: ignore
        # all len 6
        cat_feats = [
            CatTuple(*fv)._asdict()
            for fv in zip(*[enc[1][1] for enc in label_encoders])
        ]

        nlp = Spacy.get_instance(disable=["ner"])

        MAX_STRING_LEN = 200

        def get_values(patent: PatentApplication, field: str) -> list:
            val = patent.get(field)
            if isinstance(val, list):
                return [v[0:MAX_STRING_LEN] for v in val]
            if val is None:
                return [None]
            return [val[0:MAX_STRING_LEN]]

        cat_tensors: list[list[torch.Tensor]] = [
            embedding_layers[field](torch.tensor(patent[field], dtype=torch.int))
            for field in categorical_fields
            for patent in cat_feats
        ]
        max_len = max(f.size(0) for f in flatten(cat_tensors))
        padded_cat_tensors = [
            [
                F.pad(f, (0, max_len - f.size(0)))
                if f.size(0) > 0
                else torch.empty([max_len])
                for f in cat_tensors[i]
            ]
            for i in range(len(cat_tensors))
        ]
        cat_feats = [
            torch.cat(tensor_set) if len(tensor_set) > 0 else torch.Tensor([[]])
            for tensor_set in padded_cat_tensors
        ]
        text_feats = [
            [
                torch.tensor(
                    [
                        token.vector
                        for value in get_values(patent, field)
                        for token in nlp(value)
                    ],
                )
            ]
            for field in text_fields
            for patent in patents
        ]

        def get_patent_features(i: int) -> torch.Tensor:
            combo_text_feats = torch.flatten(
                (
                    torch.cat(text_feats[i])
                    if len(text_feats[i]) > 0
                    else torch.tensor([[]])
                )
            )

            combo_cat_feats = torch.flatten(cat_feats[i])

            combined = torch.cat([combo_text_feats, combo_cat_feats], dim=0)
            return combined

        embeddings = [get_patent_features(i) for i, _ in enumerate(patents)]
        return embeddings

    def __prepare_dnn_data(self, patents: Sequence[PatentApplication]) -> DnnInput:
        """
        Prepare data for DNN

        Args:
            patents (Sequence[PatentApplication]): List of patents

        Returns:
            tuple[torch.Tensor, torch.Tensor]: DNN data
        """
        embeddings = self.__get_feature_embeddings(
            patents, self.dnn_categorical_fields, self.dnn_text_fields
        )
        print("EmbeddingsX1: ", len(embeddings))
        x1 = self.__size_batch_embeddings(embeddings)
        print("X1: ", x1.size())

        y = torch.tensor([patent["approval_date"] is not None for patent in patents])
        return {"x1": x1, "y": y}

    def __size_batch_embeddings(self, embeddings: list[torch.Tensor]):
        max_len = max(e.size(0) for e in embeddings)
        padded_emb = [F.pad(f, (0, max_len - f.size(0))) for f in embeddings]
        batches = [
            torch.stack(padded_emb[i : i + self.batch_size])
            for i in range(0, len(padded_emb), self.batch_size)
        ]
        print("Batches: ", len(batches), [b.size() for b in batches])
        sized = torch.stack(batches)
        return sized

    def __prepare_gnn_input(
        self,
        patents: Sequence[PatentApplication],
    ) -> GnnInput:
        """
        Prepare inputs for GNN

        Args:
            patents (Sequence[PatentApplication]): List of patents

        Returns:
            tuple[torch.Tensor, torch.Tensor]: GNN data
        """
        # TODO: enirch with pathways, targets, disease pathways
        # GNN input features for this node
        embeddings = self.__get_feature_embeddings(patents, self.gnn_categorical_fields)
        x2 = self.__size_batch_embeddings(embeddings)

        edge_index = torch.tensor([[i, i] for i in range(len(patents))])
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
            # for bi, batch in enumerate(batches):
            for i in range(num_batches):
                batch = {k: v[i] for k, v in input_dict.items()}  # type: ignore
                logging.info("Starting batch %s out of %s", i, num_batches)
                self.optimizer.zero_grad()
                pred = self.model(batch["x1"], batch["x2"], batch["edge_index"])
                loss = self.criterion(pred, batch["y"])
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
        model = CombinedModel(100)  # TODO!!
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
    model = TrainableCombinedModel()
    model.train(patents)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 patent_pos.py\nTrians patent PoS (probability of success) model"
        )
        sys.exit()

    main()
