"""
Utils for patent eNPV model
"""

from functools import partial, reduce
from itertools import accumulate
import random
from typing import Callable, Sequence, TypeVar
import logging
from pydash import uniq
import torch
import torch.nn as nn

from data.prediction.utils import (
    ModelInputAndOutput,
    encode_and_batch_all,
    encode_and_batch_input,
    encode_quantitative_fields,
)
from typings.trials import TrialSummary

from .constants import (
    BASE_ENCODER_DIRECTORY,
    MAX_ITEMS_PER_CAT,
    QUANTITATIVE_TO_CATEGORY_FIELDS,
    InputRecord,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TRAINING_PROPORTION = 0.8


T = TypeVar("T", bound=Sequence[InputRecord | TrialSummary])


def preprocess_inputs(
    records: T, quant_to_cat_fields: list[str] = QUANTITATIVE_TO_CATEGORY_FIELDS
) -> T:
    """
    Input record preprocessing

    - shuffles
    - converts quantitative fields to categorical as desired
    """
    transformations = [
        lambda inputs: sorted(inputs, key=lambda x: random.random()),
        lambda inputs: encode_quantitative_fields(
            inputs, quant_to_cat_fields, directory=BASE_ENCODER_DIRECTORY
        ),
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
    record_cnt = input_dict.y1_true.size(0)
    split_idx = round(record_cnt * training_proportion)

    # len(v) == 0 if empty input
    split_input = {
        k: torch.split(v, split_idx) if len(v) > 0 else (torch.Tensor(), torch.Tensor())
        for k, v in input_dict.__dict__.items()
    }

    for i in range(2):
        sizes = uniq([len(v[i]) for v in split_input.values() if len(v[i]) > 0])
        if len(sizes) > 1:
            raise ValueError(
                f"Split discrepancy: {[(k, len(v[i])) for k, v in split_input.items()]}"
            )

    train_input_dict = ModelInputAndOutput(**{k: v[0] for k, v in split_input.items()})
    test_input_dict = ModelInputAndOutput(**{k: v[1] for k, v in split_input.items()})

    return train_input_dict, test_input_dict


# wrapper around encode_and_batch_input for inputs
prepare_input_data = partial(
    encode_and_batch_input,
    directory=BASE_ENCODER_DIRECTORY,
    max_items_per_cat=MAX_ITEMS_PER_CAT,
)

# wrapper around encode_and_batch_all for inputs & outputs
prepare_data = partial(
    encode_and_batch_all,
    directory=BASE_ENCODER_DIRECTORY,
    max_items_per_cat=MAX_ITEMS_PER_CAT,
)


def split_input_categories(
    preds: torch.Tensor, category_sizes: dict[str, int]
) -> list[torch.Tensor]:
    """
    For multi-categorical output, split into 1 tensor per field

    Args:
        preds (torch.Tensor): Predicted values
        category_sizes (dict[str, int]): Sizes of each category
    """
    # indexes with which to split probs into 1 tensor per field
    indices = list(accumulate(category_sizes.values(), lambda x, y: x + y))[:-1]
    preds_by_field = torch.tensor_split(preds, indices, dim=1)
    return preds_by_field


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
    preds_by_field = split_input_categories(preds, category_sizes)
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
