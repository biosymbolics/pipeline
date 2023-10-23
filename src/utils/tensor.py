"""
Tensor utilities
"""
import logging
from typing import Sequence
import torch
import torch.nn.functional as F
import numpy as np
from utils.list import BATCH_SIZE, batch

from typings.core import Primitive


def pad_or_truncate_to_size(tensor: torch.Tensor, shape: tuple[int, ...]):
    """
    Pad or truncate a tensor to a given size

    Args:
        tensor (torch.Tensor): Tensor to pad or truncate
        size (tuple[int, ...]): Size to pad or truncate to
    """
    if len(tensor) == 0:
        return torch.zeros(shape)
    # Make sure the size is correct
    if len(tensor.size()) != len(shape):
        logging.error(
            "N dims must match (expected %s vs actual %s)",
            shape,
            tensor.size(),
        )
        print("Actual", tensor)
        raise ValueError("N dims must match")

    # For each dimension
    for dim in range(len(shape)):
        # If this dimension of the tensor is smaller than required, pad it
        if tensor.size(dim) < shape[dim]:
            padding_needed = shape[dim] - tensor.size(dim)
            # Create a padding configuration with padding for this dimension only
            padding = [0] * len(shape) * 2
            padding[-(dim * 2 + 2)] = padding_needed
            tensor = F.pad(tensor, padding)
        # If this dimension of the tensor is larger than required, truncate it
        elif tensor.size(dim) > shape[dim]:
            index = [slice(None)] * len(shape)  # Start with all dimensions
            index[dim] = slice(0, shape[dim])  # Set the slice for this dimension
            tensor = tensor[tuple(index)]
    return tensor


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
    return [torch.tensor(batch) for batch in batches]


def is_array_like(data: object) -> bool:
    """
    Check if data is array-like
    """
    if isinstance(data, Sequence):
        return True
    if type(data) == np.ndarray:
        return True
    return False


def array_to_tensor(data: Sequence, shape: tuple[int, ...]) -> torch.Tensor:
    """
    Turn a list (of lists (of lists)) into a tensor

    Args:
        data (list): list (of lists (of lists))
        shape (tuple[int, ...]): Shape of the desired output tensor (TODO: automatically infer)
    """
    if not is_array_like(data):
        return torch.tensor(data)  # ??
    if len(data) <= 0:
        return torch.zeros(shape)

    # all tensors
    if all(map(lambda d: isinstance(d, torch.Tensor), data)):
        stacked = torch.stack([pad_or_truncate_to_size(d, shape[1:]) for d in data])
        return pad_or_truncate_to_size(stacked, shape)
    if is_array_like(data[0]):
        tensors = [array_to_tensor(d, shape[1:]) for d in data]
        return torch.stack(tensors)
    return pad_or_truncate_to_size(torch.tensor(data), shape)


def reverse_embedding(
    embedded_tensor: torch.Tensor, weights: list[torch.Tensor]
) -> torch.Tensor:
    """
    Reverse multi-field/multi-select embeddings

    Args:
        embedded_tensor: [seq_length, num_fields, max_selections, emb_size]
        weights: [dict_size, emb_size]
    """

    def get_encoding_idx_by_field(field_index: int):
        w = weights[field_index].unsqueeze(0)

        # outputs batch_size x selections x emb_size (1 per field)
        field_slice = embedded_tensor[:, field_index, :, :]

        # (batch_size x max selections) x emb_size
        field_select_slice = field_slice.reshape(-1, field_slice.shape[-1])

        def get_encoding_idx(i):
            dist = (field_select_slice[i] - w).abs().sum(dim=1)
            encoding_idx = dist.argmin().item()
            return encoding_idx

        distances = [get_encoding_idx(i) for i in range(field_select_slice.shape[0])]
        return torch.tensor(distances).reshape(field_slice.size(0), field_slice.size(1))

    outputs = [get_encoding_idx_by_field(i) for i in range(len(weights))]
    output = torch.stack(outputs, dim=1)
    return output
