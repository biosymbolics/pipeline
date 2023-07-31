"""
Tensor utilities
"""
import logging
import torch
import torch.nn.functional as F


def pad_or_truncate_to_size(tensor: torch.Tensor, size: tuple[int, ...]):
    """
    Pad or truncate a tensor to a given size

    Args:
        tensor (torch.Tensor): Tensor to pad or truncate
        size (tuple[int, ...]): Size to pad or truncate to
    """
    # Make sure the size is correct
    if len(tensor.size()) != len(size):
        logging.error(
            "Size must match number of dimensions in tensor (%s vs %s)",
            size,
            tensor.size(),
        )
        raise ValueError("Size must match number of dimensions in tensor")

    # For each dimension
    for dim in range(len(size)):
        # If this dimension of the tensor is smaller than required, pad it
        if tensor.size(dim) < size[dim]:
            padding_needed = size[dim] - tensor.size(dim)
            # Create a padding configuration with padding for this dimension only
            padding = [0] * len(size) * 2
            padding[-(dim * 2 + 2)] = padding_needed
            tensor = F.pad(tensor, padding)
        # If this dimension of the tensor is larger than required, truncate it
        elif tensor.size(dim) > size[dim]:
            index = [slice(None)] * len(size)  # Start with all dimensions
            index[dim] = slice(0, size[dim])  # Set the slice for this dimension
            tensor = tensor[tuple(index)]
    return tensor
