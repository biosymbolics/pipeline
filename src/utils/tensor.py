"""
Tensor utilities
"""

import logging
from typing import Sequence, cast
import torch
import torch.nn.functional as F
import numpy as np
import torch
import logging

from typings.core import Primitive
from utils.list import BATCH_SIZE, batch, is_sequence


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def pad_or_truncate_to_size(
    tensor: torch.Tensor, shape: tuple[int, ...]
) -> torch.Tensor:
    """
    Pad or truncate a tensor to a given size

    Args:
        tensor (torch.Tensor): Tensor to pad or truncate
        size (tuple[int, ...]): Size to pad or truncate to
    """
    if tensor.size() == shape:
        return tensor

    if len(tensor.size()) == 0:
        return torch.zeros(shape)

    dim_delta = len(shape) - len(tensor.size())

    # If the tensor is smaller than the required size, add dimensions
    for _ in range(dim_delta):
        tensor = tensor.unsqueeze(0)

    # or larger, remove dimensions
    if dim_delta < 0:
        for _ in range(abs(dim_delta)):
            tensor = tensor.squeeze()

    # Confirm the size is correct
    if len(tensor.size()) != len(shape):
        logging.error(
            "N dims must match (expected %s vs actual %s)", shape, tensor.size()
        )
        raise ValueError("N dims must match")

    # For each dimension
    for dim in range(len(shape)):
        # If this dimension of the tensor is smaller than required, pad it
        if tensor.size(dim) < shape[dim]:
            padding_needed = shape[dim] - tensor.size(dim)
            # Create a padding configuration with padding for this dimension only
            padding = [0] * len(shape) * 2
            padding[-(dim * 2 + 2)] = padding_needed
            tensor = F.pad(tensor, padding)  # type: ignore
        # If this dimension of the tensor is larger than required, truncate it
        elif tensor.size(dim) > shape[dim]:
            index = [slice(None)] * len(shape)  # Start with all dimensions
            index[dim] = slice(0, shape[dim])  # Set the slice for this dimension
            tensor = tensor[tuple(index)]  # type: ignore

    return tensor


def array_to_tensor(data: Sequence, shape: tuple[int, ...]) -> torch.Tensor:
    """
    Turn a list (of lists (of lists)) into a tensor

    Args:
        data (list): list (of lists (of lists))
        shape (tuple[int, ...]): Shape of the desired output tensor (TODO: automatically infer)

    Notes:
    - this is pretty brittle because it is so general
    - passing in [torch.Tensor(), torch.LongTensor()] will yield a FloatTensor
    """
    if len(shape) == 0 or shape[0] == 0:
        raise ValueError("shape must have at least 1 dimension")
    if not is_sequence(data):
        raise ValueError("data must be a sequence")
    if isinstance(data, torch.Tensor):
        return data

    is_all_scalars = all(map(lambda d: is_scalar(d), data))
    is_all_tensors = all(map(lambda d: isinstance(d, torch.Tensor), data))

    # if all tensors but also "scalars", it means they're empty (TODO: fix)
    if len(data) == 0 or (is_all_tensors and is_all_scalars):
        return torch.zeros(shape)

    if not is_all_scalars and len(shape) > 1:
        if is_all_tensors:
            stacked = torch.stack(
                [
                    pad_or_truncate_to_size(torch.Tensor(d), shape[1:])
                    for d in data
                    if not is_scalar(d)
                ]
            )
            return pad_or_truncate_to_size(stacked, shape)

        # 2d+ array
        if is_sequence(data[0]):
            tensor = torch.stack([array_to_tensor(d, shape[1:]) for d in data])
            return pad_or_truncate_to_size(tensor, shape)

    # 1d
    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
    elif isinstance(data, list):
        if all(map(lambda d: isinstance(d, int), data)):
            tensor = torch.LongTensor(data)
        else:
            try:
                tensor = torch.Tensor(data)
            except Exception as e:
                logging.error("Error converting %s to tensor", data)
                raise e
    else:
        raise ValueError(f"Unknown data type {type(data)}")

    return pad_or_truncate_to_size(tensor, shape)  # slow if np.array


def batch_as_tensors(
    items: list[Primitive], batch_size: int = BATCH_SIZE
) -> Sequence[torch.Tensor]:
    """
    Turns a list into a list of tensors of size `batch_size`

    Args:
        items (list): list to batch
        batch_size (int, optional): batch size. Defaults to BATCH_SIZE.
    """
    batches = batch(items, batch_size)
    return [torch.Tensor(batch) for batch in batches]


def is_scalar(d):
    """
    Returns True if d is a scalar (int, float, etc.)
    """
    is_tensor = isinstance(d, torch.Tensor) and (len(d.shape) == 0 or d.shape[0] == 0)
    is_numpy_scalar = isinstance(d, np.number)
    return is_tensor or is_numpy_scalar or isinstance(d, (int, float))


def l1_regularize(
    vector: torch.Tensor, sparsity_threshold: float = 0.1
) -> torch.Tensor:
    """
    l1 regularization with cutoff sparsity option
    """
    vector[vector.abs() < sparsity_threshold] = 0  # sparsify
    with torch.no_grad():
        return F.normalize(vector, p=1, dim=0)  # l1 normalize


def combine_tensors(
    a: torch.Tensor, b: torch.Tensor, a_weight: float = 0.8
) -> torch.Tensor:
    """
    Weighted combination of two vectors, regularized
    """
    b_weight = 1 - a_weight
    vector = a_weight * a + b_weight * b
    return vector


def truncated_svd(vector: torch.Tensor, variance_threshold=0.3) -> list[torch.Tensor]:
    """
    Torch implementation of TruncatedSVD
    """
    # l1 reg
    v_sparse = l1_regularize(vector, sparsity_threshold=0.3)

    # SVD
    U, S, V = torch.linalg.svd(v_sparse)

    # Sorted eigenvalues
    E = torch.sort(S**2 / torch.sum(S**2), descending=True)

    # Cumulative energy
    cum_energy = torch.cumsum(E[0], dim=0)

    mask = cum_energy > variance_threshold
    k = torch.sum(mask).int() + 1

    # Compute reduced components
    U_reduced = U[:, :k]

    t = [v.squeeze() for v in torch.split(U_reduced, 1, dim=1)]
    return t


def calc_dynamic_cutoff(scores: list[float], num_stddev=1.0) -> float:
    """
    Determine a dynamic cutoff for a list of scores based on the mean and standard deviation.

    Parameters:
    - scores (list[float]): list of scores
    - num_stddev (float): standard deviations from the mean to set as the threshold.

    Returns:
    - the cutoff score
    """
    if not scores:
        logger.error("No scores to calculate threshold")
        return 0.0

    scores_tensor = torch.tensor(scores)

    # Calculate the mean and standard deviation
    mean_score = torch.mean(scores_tensor)
    stddev_score = torch.std(scores_tensor)

    # Calculate the threshold
    threshold = mean_score - (num_stddev * stddev_score)

    return threshold.item()


def similarity_with_residual_penalty(
    a: torch.Tensor,
    b: torch.Tensor,
    alpha: float = 0.45,
    name: str = "na",
) -> float:
    """
    Compute a weighted similarity score that penalizes a large residual.

    Args:
        a (torch.Tensor): tensor a
        b (torch.Tensor): tensor b
        distance (float, optional): cosine/"angular" distance. Defaults to None, in which case it is computed.
        alpha (float, optional): 1 - a == weight of the residual penalty. Defaults to 0.5.
    """
    with torch.no_grad():
        similarity = 2 - F.cosine_similarity(a, b, dim=0)

        # Compute residual
        residual = torch.subtract(a, b)

        # Frobenius norm
        residual_norm = torch.norm(residual, p="fro")

        # Scale residual norm to [0,1] range
        scaled_residual_norm = torch.divide(residual_norm, torch.norm(a))

        # Weighted score
        score = alpha * similarity + (1 - alpha) * (1 - torch.abs(scaled_residual_norm))
        logger.debug(
            "%s: Similarity %s, Residual: %s, Score: %s",
            name,
            similarity.item(),
            scaled_residual_norm.item(),
            score.item(),
        )
        return score.item()


def tensor_mean(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Takes a list of Nx0 vectors and returns the mean vector (Nx0)
    """
    return torch.stack(list(vectors), dim=1).mean(dim=1)
