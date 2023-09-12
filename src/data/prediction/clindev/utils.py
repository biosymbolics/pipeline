"""
Utils for patent eNPV model
"""

from typing import Sequence, cast
import logging

from torch import unsqueeze

from data.prediction.utils import (
    batch_and_pad,
    get_feature_embeddings,
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
    categorical_fields: list[str],
    text_fields: list[str],
    y1_categorical_fields: list[str],  # e.g. randomization_type
    y2_field: str,
) -> DnnInput:
    """
    Prepare data for DNN
    """
    logger.info("Preparing inputs for DNN")
    embeddings = get_feature_embeddings(trials, categorical_fields, text_fields)  # type: ignore
    x1 = resize_and_batch(embeddings, batch_size)

    y1_vals = get_feature_embeddings(trials, y1_categorical_fields, [])  # type: ignore
    y1 = batch_and_pad(cast(list[Primitive], y1_vals), batch_size)

    y2_vals = [float(trial[y2_field]) for trial in trials]
    print("Y2 vals", y2_vals[0:100])
    y2 = unsqueeze(batch_and_pad(cast(list[Primitive], y2_vals), batch_size), 2)
    logger.info(
        "X1: %s, Y1: %s, Y2: %s",
        x1.size(),
        y1.size(),
        y2.size(),
    )
    return {"x1": x1, "y1": y1, "y2": y2}
