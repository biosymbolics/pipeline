import os
from typing import Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv
import polars as pl

from typings.patents import PatentApplication

LR = 1e-3  # learning rate
CHECKPOINT_PATH = "patent_model_checkpoints"
OPTIMIZER_CLASS = torch.optim.Adam
SAVE_FREQUENCY = 5  # save model every 5 epochs

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
        dnn_input_dim: int = 256,
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


def get_model() -> tuple[CombinedModel, torch.optim.Optimizer]:
    """
    Get model and optimizer

    Returns:
        tuple[CombinedModel, torch.optim.Optimizer]: Model and optimizer
    """
    model = CombinedModel()
    optimizer = OPTIMIZER_CLASS(model.parameters(), lr=LR)
    return (model, optimizer)


def save_checkpoint(model: CombinedModel, optimizer: torch.optim.Optimizer, epoch: int):
    """
    Save model checkpoint

    Args:
        model (CombinedModel): Model to save
        optimizer (torch.optim.Optimizer): Optimizer to save
        epoch (int): Epoch number
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
    }
    checkpoint_name = f"checkpoint_{epoch}.pt"
    torch.save(checkpoint, os.path.join(CHECKPOINT_PATH, checkpoint_name))


Data = list


def do_train(
    data: Data,
    model: Optional[CombinedModel] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    start_epoch: int = 0,
    num_epochs: int = 20,
) -> CombinedModel:
    """
    Train model

    Args:
        data (Data): List of data batches
        model (Optional[CombinedModel], optional): Model to train. Defaults to None.
        optimizer (Optional[torch.optim.Optimizer], optional): Optimizer to use. Defaults to None.
        start_epoch (int, optional): Epoch to start training from. Defaults to 0.
        num_epochs (int, optional): Number of epochs to train for. Defaults to 20.
    """
    model, optimizer = (model, optimizer) if model and optimizer else get_model()
    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(start_epoch, num_epochs):
        for batch in data:
            if epoch % SAVE_FREQUENCY == 0:
                save_checkpoint(model, optimizer, epoch)
            optimizer.zero_grad()
            pred = model(batch.x1, batch.x2, batch.edge_index)
            loss = criterion(pred, batch.y)
            loss.backward()
            optimizer.step()

    return model


def resume(data: Data, checkpoint_name: str):
    """
    Resume training from a checkpoint

    Args:
        data (list): List of data batches
        checkpoint_name (str): Checkpoint from which to resume
    """
    model = CombinedModel()
    checkpoint_file = os.path.join(CHECKPOINT_PATH, checkpoint_name)

    if not os.path.exists(checkpoint_file):
        raise Exception(f"Checkpoint {checkpoint_name} does not exist")

    checkpoint = torch.load(checkpoint_file)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = OPTIMIZER_CLASS(model.parameters(), lr=LR)  # Initialize new optimizer
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    return do_train(data, model, optimizer, start_epoch)


DNN_PATENT_FEATURES = [
    "application_kind",
    "assignees",
    "attributes",
    "compounds",
    "country",
    "diseases",
    "ipc_codes",
    "inventors",
    "mechanisms",
]  # title and abstract??


def get_features(patent: PatentApplication, feature_fields: list[str]) -> list:
    """
    Get features for a patent

    Args:
        patent (PatentApplication): Patent

    Returns:
        list: List of features
    """
    features = [patent[feature] for feature in feature_fields]
    return features


def get_feature_embeddings(patents: list[PatentApplication], feature_fields: list[str]):
    df = pl.from_dicts([patent.__dict__ for patent in patents])
    size_map = dict(
        [(field, df.select(pl.col(field)).n_unique()) for field in feature_fields]
    )
    embedding_layers = dict(
        [(field, nn.Embedding(size_map[field], 32)) for field in feature_fields]
    )

    def get_node_features(patent):
        return torch.cat(
            [embedding_layers[field](patent[field]) for field in feature_fields], dim=1
        )

    embeddings = [get_node_features(patent) for patent in patents]
    return embeddings


def prepare_dnn_data(
    patents: list[PatentApplication], outcome_map: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare data for DNN

    Args:
        patents (list[PatentApplication]): List of patents

    Returns:
        tuple[torch.Tensor, torch.Tensor]: DNN data
    """
    embeddings = get_feature_embeddings(patents, DNN_PATENT_FEATURES)
    x1 = torch.cat(embeddings, dim=1)

    things_patented = [
        " + ".join(patent["compounds"] + patent["mechanisms"]) for patent in patents
    ]

    y = torch.tensor(
        [outcome_map.get(patented_thing, 0) for patented_thing in things_patented]
    )
    return (x1, y)


def prepare_gnn_data(
    patents: list[PatentApplication],
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare data for GNN

    Args:
        patents (list[PatentApplication]): List of patents

    Returns:
        tuple[torch.Tensor, torch.Tensor]: GNN data
    """
    GNN_FEATURES = [
        "diseases",
        "compounds",
        "mechanisms",
    ]  # TODO: enirch with pathways, targets, disease pathways

    # GNN input features for this node
    embeddings = get_feature_embeddings(patents, GNN_FEATURES)
    x2 = torch.cat(embeddings, dim=1)

    edge_index = torch.tensor([[i, i] for i in range(len(patents))])
    return (x2, edge_index)
