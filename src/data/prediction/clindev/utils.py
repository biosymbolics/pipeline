"""
Utils for patent eNPV model
"""

from typing import Sequence, cast
import logging
import torch
import torch.nn.functional as F

from data.prediction.utils import (
    batch_and_pad,
    vectorize_features,
    resize_and_batch,
)
from typings.core import Primitive
from typings.trials import TrialSummary
from utils.tensor import reduce_last_dim

from .types import DnnInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def prepare_inputs(
    trials: Sequence[TrialSummary],
    batch_size: int,
    categorical_fields: list[str],
    text_fields: list[str],
    y1_categorical_fields: list[str],  # e.g. randomization
    y2_field: str,
) -> tuple[DnnInput, list[torch.Tensor], dict[str, int]]:
    """
    Prepare data for DNN
    """
    logger.info("Preparing inputs for DNN")
    vectorized_feats = vectorize_features(trials, categorical_fields, text_fields)  # type: ignore

    # (batches) x (seq length) x (num cats) x (max items per cat) x (embed size)
    x1 = resize_and_batch(vectorized_feats.embeddings, batch_size)

    y1_field_indexes = tuple(n for n in range(len(y1_categorical_fields)))
    # (batches) x (seq length) x (num cats) x (max items per cat) x (embed size)
    y1 = x1[:, :, y1_field_indexes, :]

    # (batches) x (seq length) x (num cats) x (items per cat)
    y1_labels_by_cat = vectorized_feats.encodings[:, y1_field_indexes, :]

    y1_categories = resize_and_batch(y1_labels_by_cat, batch_size)
    y2_vals = [float(trial[y2_field]) for trial in trials]

    # (batches) x (seq length) x 1
    y2 = torch.unsqueeze(batch_and_pad(cast(list[Primitive], y2_vals), batch_size), 2)
    logger.info(
        "x1: %s, y1: %s, y1_cats: %s, Y2: %s",
        x1.size(),
        y1.size(),
        y1_categories.size(),
        y2.size(),
    )

    return (
        {
            "x1": x1,
            "y1": y1,
            "y1_categories": y1_categories,
            "y2": y2,
        },
        vectorized_feats.embedding_weights,
        vectorized_feats.category_size_map,
    )
