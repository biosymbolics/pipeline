"""
Utils for patent eNPV model
"""

from functools import reduce
from itertools import accumulate
import random
from typing import Callable, Sequence
import logging
import torch
import torch.nn as nn

from data.prediction.utils import (
    encode_and_batch_features,
    encode_quantitative_fields,
    resize_and_batch,
    encode_single_select_categories as vectorize_single_select,
)
from typings.trials import TrialSummary

from .constants import DEVICE, MAX_ITEMS_PER_CAT
from .types import AllCategorySizes, DnnInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def preprocess_inputs(records: Sequence[TrialSummary], quant_to_cat_fields: list[str]):
    """
    Input record preprocessing

    - shuffles
    - converts quantitative fields to categorical as desired
    """
    transformations = [
        lambda inputs: sorted(inputs, key=lambda x: random.random()),
        lambda inputs: encode_quantitative_fields(inputs, quant_to_cat_fields),
    ]
    output = reduce(lambda x, f: f(x), transformations, records)

    return output


def prepare_inputs(
    trials: Sequence[TrialSummary],
    batch_size: int,
    single_select_categorical_fields: list[str],
    multi_select_categorical_fields: list[str],
    text_fields: list[str],
    quantitative_fields: list[str],
    y1_categorical_fields: list[str],  # e.g. cat fields to predict, e.g. randomization
    y2_field: str,
    device: str = DEVICE,
) -> tuple[DnnInput, AllCategorySizes]:
    """
    Prepare data for DNN
    """
    logger.info("Preparing inputs for DNN")

    batched_feats, sizes = encode_and_batch_features(
        batch_size,
        trials,  # type: ignore
        single_select_categorical_fields,
        multi_select_categorical_fields,
        text_fields,
        quantitative_fields,
        flatten_batch=False,
        max_items_per_cat=MAX_ITEMS_PER_CAT,
        device=device,
    )

    # (batches) x (seq length) x (num cats) x (items per cat)
    y1_size_map, vectorized_y1 = vectorize_single_select(trials, y1_categorical_fields, device=device)  # type: ignore
    y1 = resize_and_batch(vectorized_y1, batch_size)
    y1_categories = resize_and_batch(vectorized_y1, batch_size)
    y2 = resize_and_batch(
        torch.Tensor([trial[y2_field] for trial in trials]).to(device), batch_size
    )

    # (batches) x (seq length) x 1
    # y2 = torch.unsqueeze(batch_and_pad(y2_vals, batch_size), 2).to(device)
    logger.info(
        "y1: %s, y1_categories: %s, y2: %s",
        y1.size(),
        y1_categories.size(),
        y2.size(),
    )

    return (
        {
            "multi_select_x": batched_feats.multi_select,
            "single_select_x": batched_feats.single_select,
            "text_x": batched_feats.text,
            "quantitative_x": batched_feats.quantitative,
            "y1": y1,
            "y1_categories": y1_categories,
            "y2": y2,
        },
        AllCategorySizes(*sizes, y1=y1_size_map),  # type: ignore
    )


def calc_categories_loss(
    y1_probs: torch.Tensor,
    y1_true: torch.Tensor,
    category_sizes: dict[str, int],
    criterion: Callable,
) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
    """
    For multi-categorical output, add up loss across categories
    """

    # indexes with which to split y1_probs into 1 tensor per field
    indices = list(accumulate(category_sizes.values(), lambda x, y: x + y))[:-1]
    y1_probs_by_field = torch.tensor_split(y1_probs, indices, dim=1)
    y1_true_by_field = [y1.squeeze() for y1 in torch.split(y1_true, 1, dim=1)]
    loss = torch.stack(
        [
            criterion(y1_by_field.float(), y1_true_set)
            for y1_by_field, y1_true_set in zip(
                y1_probs_by_field,
                y1_true_by_field,
            )
        ]
    ).sum()
    return loss, y1_probs_by_field, y1_true_by_field


def embed_cat_inputs(
    input: list[torch.Tensor], embeddings: nn.ModuleDict, device: str
) -> torch.Tensor:
    """
    Embed categorical inputs

    Args:
        input (list[torch.Tensor]): List of categorical inputs as encodings/tensors
        embeddings (nn.ModuleDict): Embedding layers
        device (str): Device to use
    """
    cats_input = torch.cat(
        [
            el(x)
            for x, el in zip(
                input,
                embeddings.values(),
            )
        ],
        dim=1,
    ).to(device)
    cats_input = cats_input.view(*cats_input.shape[0:1], -1)
    return cats_input
