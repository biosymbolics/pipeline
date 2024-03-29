"""
Patent Probability of Success (PoS) model(s)
"""
from typing import Any, Optional
import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv


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
        >>> model = CombinedModel()
        >>> checkpoint = torch.load("patent_model_checkpoints/checkpoint_15.pt", map_location=torch.device('mps'))
        >>> model.load_state_dict(checkpoint["model_state_dict"])
        >>> model.eval()
        >>> torch.save(model, 'patent-model.pt')
    """

    def __init__(
        self,
        dnn_input_dim: int,
        gnn_input_dim: int,
        dnn_hidden_dim: int = 128,
        gnn_hidden_dim: int = 128,
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
