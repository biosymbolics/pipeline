"""
Utils for patent eNPV model
"""

from typing import Sequence, cast
import logging

from data.prediction.utils import (
    batch_and_pad,
    get_feature_embeddings,
    resize_and_batch,
)
from typings.core import Primitive
from typings.trials import TrialSummary

from .types import DnnInput


def prepare_inputs(
    trials: Sequence[TrialSummary],
    batch_size: int,
    dnn_categorical_fields: list[str],
    dnn_text_fields: list[str],
) -> DnnInput:
    """
    Prepare data for DNN
    """
    embeddings = get_feature_embeddings(trials, dnn_categorical_fields, dnn_text_fields)  # type: ignore
    x1 = resize_and_batch(embeddings, batch_size)

    y_vals = [trial["end_date"] - trial["start_date"] for trial in trials]
    y = batch_and_pad(cast(list[Primitive], y_vals), batch_size)
    logging.info(
        "X1: %s, Y: %s (t: %s, f: %s)",
        x1.size(),
        y.size(),
        len([y for y in y_vals if y == 1.0]),
        len([y for y in y_vals if y == 0.0]),
    )
    return {"x1": x1, "y": y}
