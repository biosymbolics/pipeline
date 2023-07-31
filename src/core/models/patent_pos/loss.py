"""
Loss functions for patent_pos model.
"""
import logging
import torch
from torch.nn import functional as F


def contrastive_loss(
    output1: torch.Tensor,
    output2: torch.Tensor,
    label: torch.Tensor,
    margin: float = 1.0,
):
    """
    Contrastive loss function.

    loss = (1/2N) * sum(y * d^2 + (1 - y) * max(margin - d, 0)^2)

    N == batch size

    Args:
        output1 (torch.Tensor): output from the first network
        output2 (torch.Tensor): output from the second network
        label (torch.Tensor): label for the pair

    Usage:
    ```
        y_true_sample = pred[torch.where(y == 1.0)[0]]
        y_false_sample = pred[torch.where(y == 0.0)[0]]
        loss = contrastive_loss(y_true_sample, y_false_sample, y)
    ```

    TODO:
        - multi-stage loss: patent -> trial, trial -> regulatory approval
    """
    logging.info(
        "output1: %s, output2: %s, label: %s",
        output1.size(),
        output2.size(),
        label.size(),
    )
    euclidean_distance = F.pairwise_distance(output1, output2)
    loss_contrastive = torch.mean(
        (1 - label) * torch.pow(euclidean_distance, 2)
        + (label) * torch.pow(torch.clamp(margin - euclidean_distance, min=0.0), 2)
    )

    return loss_contrastive
