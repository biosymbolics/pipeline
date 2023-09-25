"""
Utils for patent eNPV model
"""

from typing import Sequence, cast
import logging
import torch

from data.prediction.utils import (
    batch_and_pad,
    encode_and_batch_features,
    resize_and_batch,
    encode_single_select_categories as vectorize_single_select,
)
from typings.core import Primitive
from typings.trials import TrialSummary

from .constants import DEVICE
from .types import AllCategorySizes, DnnInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        device=device,
    )

    # (batches) x (seq length) x (num cats) x (items per cat)
    y1_size_map, vectorized_y1 = vectorize_single_select(trials, y1_categorical_fields, device=device)  # type: ignore
    y1 = resize_and_batch(vectorized_y1, batch_size)
    y1_categories = resize_and_batch(vectorized_y1, batch_size)  # .squeeze()
    y2_vals = [float(trial[y2_field]) for trial in trials]

    # (batches) x (seq length) x 1
    y2 = torch.unsqueeze(
        batch_and_pad(cast(list[Primitive], y2_vals), batch_size), 2
    ).to(device)
    logger.info(
        "y1: %s, y1_categories: %s, Y2: %s",
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
