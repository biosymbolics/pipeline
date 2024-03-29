"""
Utils for patent eNPV model
"""

from typing import Sequence, cast
import logging
import torch

from models.utils import (
    batch_and_pad,
    encode_features,
    resize_and_batch,
)
from typings.core import Primitive
from typings.documents.patents import ScoredPatent as PatentApplication

from .types import AllInput, ModelInput, GnnInput


def prepare_inputs(
    patents: Sequence[PatentApplication],
    batch_size: int,
    dnn_categorical_fields: list[str],
    dnn_text_fields: list[str],
    gnn_categorical_fields: list[str],
) -> AllInput:
    """
    Prepare inputs for model
    """

    def _prepare_dnn_data(
        patents: Sequence[PatentApplication],
        dnn_categorical_fields: list[str],
        dnn_text_fields: list[str],
        batch_size: int,
    ) -> ModelInput:
        """
        Prepare data for DNN
        """
        embeddings, _ = encode_features(
            patents, dnn_categorical_fields, dnn_text_fields  # type: ignore
        )
        x1 = resize_and_batch(embeddings, batch_size)

        y_vals = [
            0.0
            for patent in patents  # (1.0 if patent.approval_date is not None else 0.0) for patent in patents
        ]
        y = batch_and_pad(cast(list[Primitive], y_vals), batch_size)
        logging.info(
            "X1: %s, Y: %s (t: %s, f: %s)",
            x1.size(),
            y.size(),
            len([y for y in y_vals if y == 1.0]),
            len([y for y in y_vals if y == 0.0]),
        )
        return {"x1": x1, "y": y}

    def _prepare_gnn_input(
        patents: Sequence[PatentApplication],
        gnn_categorical_fields: list[str],
        batch_size: int,
    ) -> GnnInput:
        """
        Prepare inputs for GNN
        """
        # TODO: enirch with pathways, targets, disease pathways
        embeddings, _ = encode_features(patents, gnn_categorical_fields)  # type: ignore
        x2 = resize_and_batch(embeddings, batch_size)
        edge_index = [torch.Tensor([i, i]) for i in range(len(patents))]
        ei = batch_and_pad(edge_index, batch_size)
        # logging.info("X2: %s, EI %s", x2.size(), ei.size())

        return {"x2": x2, "edge_index": ei}

    return cast(
        AllInput,
        {
            **_prepare_dnn_data(
                patents, dnn_categorical_fields, dnn_text_fields, batch_size
            ),
            **_prepare_gnn_input(patents, gnn_categorical_fields, batch_size),
        },
    )
