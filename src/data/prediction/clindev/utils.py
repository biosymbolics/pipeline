"""
Utils for patent eNPV model
"""

from typing import Any, Callable, Sequence, cast
import logging
from typing_extensions import TypeVar
import torch

from data.prediction.utils import (
    batch_and_pad,
    vectorize_and_batch_features,
    resize_and_batch,
    vectorize_single_select_categories as vectorize_single_select,
)
from typings.core import Primitive
from typings.trials import TrialSummary

from .constants import EMBEDDING_DIM
from .types import DnnInput

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
    embedding_dim: int = EMBEDDING_DIM,
    flatten_batch: bool = False,
) -> tuple[DnnInput, dict[str, int]]:
    """
    Prepare data for DNN
    """
    logger.info("Preparing inputs for DNN")

    batched_feats = vectorize_and_batch_features(
        batch_size,
        trials,  # type: ignore
        single_select_categorical_fields,
        multi_select_categorical_fields,
        text_fields,
        quantitative_fields,
        embedding_dim=embedding_dim,
        flatten_batch=flatten_batch,
    )

    # (batches) x (seq length) x (num cats) x (items per cat)
    vectorized_y1 = vectorize_single_select(trials, y1_categorical_fields, embedding_dim)  # type: ignore
    y1 = resize_and_batch(vectorized_y1.embeddings, batch_size)
    y1_categories = resize_and_batch(vectorized_y1.encodings.squeeze(), batch_size)
    y2_vals = [float(trial[y2_field]) for trial in trials]

    # (batches) x (seq length) x 1
    y2 = torch.unsqueeze(batch_and_pad(cast(list[Primitive], y2_vals), batch_size), 2)
    logger.info(
        "y1: %s, y1_categories: %s, Y2: %s",
        y1.size(),
        y1_categories.size(),
        y2.size(),
    )

    return (
        {
            "multi_select_x": batched_feats.multi_select.to("mps"),
            "single_select_x": batched_feats.single_select.to("mps"),
            "text_x": batched_feats.text.to("mps"),
            "quantitative_x": batched_feats.quantitative.to("mps"),
            "y1": y1.to("mps"),
            "y1_categories": y1_categories.to("mps"),
            "y2": y2.to("mps"),
        },
        vectorized_y1.category_size_map,
    )
