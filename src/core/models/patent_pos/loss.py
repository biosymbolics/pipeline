"""
Loss functions for patent_pos model.
"""
import logging
import torch
from torch import nn
from torch.nn import functional as F


class FocalLoss(nn.Module):
    """
    Focal loss is designed to address classification imbalance by down-weighting inliers
    (easy examples) such that their contribution to the total loss is small even if their
    number is large. (from GPT4)

    OG paper: https://arxiv.org/pdf/1708.02002.pdf
    """

    def __init__(self, alpha: float = 0.25, gamma: int = 2, reduce: bool = True):
        """
        alpha (float: range [0, 1]): factor to balance the relative importance of positive/negative examples
        gamma (int): a focusing parameter that controls how much the loss focuses on harder examples.
               The larger the gamma, the more the loss penalizes (significantly) false negatives.
        reduce (bool): if True, calculate the mean loss over the batch.
                If False, return the loss for each example in the batch.
        """

        if gamma <= 0:
            raise ValueError("gamma should be a positive integer")

        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduce = reduce

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        BCE_loss = F.binary_cross_entropy_with_logits(x, y, reduce=False)

        """
        pt == probability assigned to the true class by the model.
        it's an algebraic trick to recover this from BCE_loss, given
        BCE_loss = -log(pt) for y = 1
        therefore, pt = exp(-BCE_loss) for y = 1
        """
        pt = torch.exp(-BCE_loss)

        """
        Calculate Focal Loss as defined in the original paper.
        It modifies the BCE_loss by a factor that gives more weight to hard examples
        F_loss increases for examples where pt is small (misclassified examples)

        (1 - pt)^gamma -> the effect of the hard examples (pt is small) is increased
        relative to the easy ones (pt is closer to 1).
        """
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            # return mean loss over batch
            return torch.mean(F_loss)

        # return per-element losses
        return F_loss
