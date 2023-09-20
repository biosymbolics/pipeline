"""
Utils for patent eNPV model
"""

from typing import Sequence, cast
import logging
import torch

from data.prediction.utils import (
    batch_and_pad,
    vectorize_features,
    resize_and_batch,
)
from typings.core import Primitive
from typings.trials import TrialSummary

from .types import DnnInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def prepare_inputs(
    trials: Sequence[TrialSummary],
    batch_size: int,
    single_select_categorical_fields: list[str],
    multi_select_categorical_fields: list[str],
    text_fields: list[str],
    y1_categorical_fields: list[str],  # e.g. cat fields to predict, e.g. randomization
    y2_field: str,
) -> tuple[DnnInput, list[torch.Tensor]]:
    """
    Prepare data for DNN
    """
    logger.info("Preparing inputs for DNN")

    if not all(
        [field in single_select_categorical_fields for field in y1_categorical_fields]
    ):
        raise ValueError(
            "y1_categorical_fields must be a subset of single_select_categorical_fields"
        )

    y1_field_indexes = tuple(n for n in range(len(y1_categorical_fields)))

    vectorized_feats = vectorize_features(
        trials,  # type: ignore
        single_select_categorical_fields,
        multi_select_categorical_fields,
        text_fields,
    )

    # (batches) x (seq length) x (num cats) x (max items per cat) x (embed size)
    multi_select_x = resize_and_batch(
        vectorized_feats.multi_select_embeddings, batch_size
    )

    single_select_x = resize_and_batch(
        vectorized_feats.single_select_embeddings, batch_size
    )

    text_x = (
        resize_and_batch(vectorized_feats.text_embeddings, batch_size)  # type: ignore
        if vectorized_feats.text_embeddings is not None
        else None
    )

    # (batches) x (seq length) x (num cats) x (max items per cat) x (embed size)
    y1 = single_select_x[:, :, y1_field_indexes, :]

    # (batches) x (seq length) x (num cats) x (items per cat)
    y1_labels_by_cat = vectorized_feats.encodings[:, y1_field_indexes, :]

    y1_categories = resize_and_batch(y1_labels_by_cat, batch_size)
    y2_vals = [float(trial[y2_field]) for trial in trials]

    # (batches) x (seq length) x 1
    y2 = torch.unsqueeze(batch_and_pad(cast(list[Primitive], y2_vals), batch_size), 2)
    logger.info(
        "multi_select_x: %s, y1: %s, y1_cats: %s, Y2: %s",
        multi_select_x.size(),
        y1.size(),
        y1_categories.size(),
        y2.size(),
    )

    return (
        {
            "multi_select_x": multi_select_x,
            "single_select_x": single_select_x,
            "text_x": text_x,
            "y1": y1,
            "y1_categories": y1_categories,
            "y2": y2,
        },
        vectorized_feats.embedding_weights,
    )
