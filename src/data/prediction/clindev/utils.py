"""
Utils for patent eNPV model
"""

from functools import reduce
from itertools import accumulate
import random
from typing import Callable, Sequence, cast
import logging
import torch
import torch.nn as nn

from data.prediction.utils import (
    ModelInput,
    ModelInputAndOutput,
    encode_and_batch_features,
    encode_quantitative_fields,
    resize_and_batch,
    encode_single_select_categories as vectorize_single_select,
)
from data.types import FieldLists, InputFieldLists
from typings.trials import TrialSummary

from .constants import DEVICE, MAX_ITEMS_PER_CAT, InputRecord
from .types import AllCategorySizes

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TRAINING_PROPORTION = 0.8


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


def split_train_and_test(
    input_dict: ModelInputAndOutput, training_proportion: float = TRAINING_PROPORTION
) -> tuple[ModelInputAndOutput, ModelInputAndOutput]:
    """
    Split out training and test data

    Args:
        input_dict (ModelInputAndOutput): Input data
        training_proportion (float): Proportion of data to use for training
    """
    total_records = input_dict.y1_true.size(0)
    split_idx = round(total_records * training_proportion)
    training_input_dict = ModelInputAndOutput(
        **{k: torch.split(v, split_idx)[0] for k, v in input_dict._asdict().items()}
    )
    test_input_dict = ModelInputAndOutput(
        **{
            k: torch.split(v, split_idx)[1] if len(v) == total_records else v
            for k, v in input_dict._asdict().items()
        }
    )

    return training_input_dict, test_input_dict


def prepare_input_data(
    records: Sequence[InputRecord],
    input_field_lists: InputFieldLists,
    batch_size: int,
    device: str,
) -> ModelInput:
    """
    Prepare input data
    """
    # encode_and_batch_features, max_items_per_cat=MAX_ITEMS_PER_CAT
    inputs, _ = encode_and_batch_features(
        records,
        field_lists=input_field_lists,
        batch_size=batch_size,
        max_items_per_cat=MAX_ITEMS_PER_CAT,
        device=device,
    )

    return inputs


def prepare_data(
    trials: Sequence[TrialSummary],
    field_lists: FieldLists,
    batch_size: int,
    device: str = DEVICE,
) -> tuple[ModelInputAndOutput, AllCategorySizes, int]:
    """
    Prepare data for model
    """
    logger.info("Preparing inputs for model")

    records = cast(Sequence[InputRecord], trials)

    batched_feats, sizes = encode_and_batch_features(
        records,
        field_lists=field_lists,
        batch_size=batch_size,
        max_items_per_cat=MAX_ITEMS_PER_CAT,
        device=device,
    )

    # (batches) x (seq length) x (num cats) x (items per cat)
    y1_size_map, vectorized_y1 = vectorize_single_select(trials, field_lists.y1_categorical, device=device)  # type: ignore
    y1 = resize_and_batch(vectorized_y1, batch_size)
    y1_categories = resize_and_batch(vectorized_y1, batch_size)
    all_y2 = (
        torch.Tensor([trial[field_lists.y2] for trial in trials])
        .type(torch.int64)
        .to(device)
    )
    y2 = resize_and_batch(all_y2, batch_size).squeeze()
    y2_oh = resize_and_batch(
        torch.zeros(all_y2.size(0), 10).to(device).scatter_(1, all_y2, 1), batch_size
    )

    # (batches) x (seq length) x 1
    logger.info(
        "y1: %s, y1_categories: %s, y2: %s, y2_oh: %s",
        y1.size(),
        y1_categories.size(),
        y2.size(),
        y2_oh.size(),
    )

    return (
        ModelInputAndOutput(
            *batched_feats,
            y1_true=y1,
            y1_categories=y1_categories,
            y2_true=y2,
            y2_oh_true=y2_oh
        ),
        AllCategorySizes(*sizes, y1=y1_size_map),  # type: ignore
        round(len(batched_feats.multi_select) / batch_size),
    )


def split_categories(
    preds: torch.Tensor,
    trues: torch.Tensor,
    category_sizes: dict[str, int],
) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
    """
    For multi-categorical output, split into 1 tensor per field

    Args:
        preds (torch.Tensor): Predicted values
        trues (torch.Tensor): True values
        category_sizes (dict[str, int]): Sizes of each category
    """
    # indexes with which to split probs into 1 tensor per field
    indices = list(accumulate(category_sizes.values(), lambda x, y: x + y))[:-1]
    preds_by_field = torch.tensor_split(preds, indices, dim=1)
    trues_by_field = [t.squeeze() for t in torch.split(trues, 1, dim=1)]
    return preds_by_field, trues_by_field


def calc_categories_loss(
    y1_probs_by_field: list[torch.Tensor],
    y1_true_by_field: list[torch.Tensor],
    criterion: Callable,
) -> torch.Tensor:
    """
    For multi-categorical output, add up loss across categories
    """
    loss = torch.stack(
        [
            criterion(y1_by_field.float(), y1_true_set)
            for y1_by_field, y1_true_set in zip(
                y1_probs_by_field,
                y1_true_by_field,
            )
        ]
    ).sum()
    return loss


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
