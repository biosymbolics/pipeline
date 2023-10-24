"""
Utils for patent eNPV model
"""

from functools import partial, reduce
from itertools import accumulate
import random
from typing import Callable, Sequence, TypeVar, cast
import logging
import torch
import torch.nn as nn
from data.prediction.types import (
    ModelInput,
    ModelInputAndOutput,
    is_model_input,
    is_model_input_output,
)

from data.prediction.utils import (
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
            criterion(y1_by_field, y1_true_set)
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


BATCH_TRANSFORMATIONS = {
    "multi_select": lambda x: torch.split(x, 1, dim=1),
    "single_select": lambda x: torch.split(x, 1, dim=1),
}


IT = TypeVar("IT", bound=ModelInputAndOutput | ModelInput)


def get_batch(
    i: int,
    input: IT,
    transformations: dict[str, Callable] = BATCH_TRANSFORMATIONS,
) -> IT | None:
    """
    Get input for batch i

    Args:
        i (int): Batch index
        input (ModelInputAndOutput): Input dict
        transformations (dict[str, Callable], optional): Transformations to apply to each field. Defaults to BATCH_TRANSFORMATIONS
    """
    if not all([len(v) > i or len(v) == 0 for v in input.values()]):
        logger.error("Batch %s is empty", i)
        return None

    def transform_batch(f: str, v: torch.Tensor, i: int):
        transform = transformations.get(f) or (lambda x: x)
        return transform(v[i]) if len(v) > 0 else torch.Tensor()

    batch = {f: transform_batch(f, v, i) for f, v in input.items()}

    if is_model_input(input):
        batch = ModelInput(**batch)
    elif is_model_input_output(input):
        batch = ModelInputAndOutput(**batch)
    else:
        raise ValueError(f"Unknown input type {type(input)}")

    return cast(IT, batch)
