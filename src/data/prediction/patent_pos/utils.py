"""
Utils for patent eNPV model
"""

from typing import Sequence, cast
import logging
import torch

from data.prediction.patent_pos.types import AllInput, DnnInput, GnnInput
from data.prediction.utils import (
    batch_and_pad,
    vectorize_features,
    resize_and_batch,
)
from typings.core import Primitive
from typings.patents import ApprovedPatentApplication as PatentApplication


# Query for approval data
# product p, active_ingredient ai, synonyms syns, approval a
"""
select
    p.ndc_product_code as ndc,
    (array_agg(distinct p.generic_name))[1] as generic_name,
    (array_agg(distinct p.product_name))[1] as brand_name,
    (array_agg(distinct p.marketing_status))[1] as status,
    (array_agg(distinct active_ingredient_count))[1] as active_ingredient_count,
    (array_agg(distinct route))[1] as route,
    (array_agg(distinct s.name)) as substance_names,
    (array_agg(distinct a.type)) as approval_types,
    (array_agg(distinct a.approval)) as approval_dates,
    (array_agg(distinct a.applicant)) as applicants
from structures s
LEFT JOIN approval a on a.struct_id=s.id
LEFT JOIN active_ingredient ai on ai.struct_id=s.id
LEFT JOIN product p on p.ndc_product_code=ai.ndc_product_code
LEFT JOIN synonyms syns on syns.id=s.id
where (syns.name ilike '%elexacaftor%' or p.generic_name ilike '%elexacaftor%' or p.product_name ilike '%elexacaftor%')
group by p.ndc_product_code;
"""


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

    def __prepare_dnn_data(
        patents: Sequence[PatentApplication],
        dnn_categorical_fields: list[str],
        dnn_text_fields: list[str],
        batch_size: int,
    ) -> DnnInput:
        """
        Prepare data for DNN
        """
        embeddings = vectorize_features(
            patents, dnn_categorical_fields, dnn_text_fields  # type: ignore
        )
        x1 = resize_and_batch(embeddings, batch_size)

        y_vals = [
            (1.0 if patent["approval_date"] is not None else 0.0) for patent in patents
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

    def __prepare_gnn_input(
        patents: Sequence[PatentApplication],
        gnn_categorical_fields: list[str],
        batch_size: int,
    ) -> GnnInput:
        """
        Prepare inputs for GNN
        """
        # TODO: enirch with pathways, targets, disease pathways
        embeddings = vectorize_features(patents, gnn_categorical_fields)  # type: ignore
        x2 = resize_and_batch(embeddings, batch_size)
        edge_index = [torch.tensor([i, i]) for i in range(len(patents))]
        ei = batch_and_pad(edge_index, batch_size)
        # logging.info("X2: %s, EI %s", x2.size(), ei.size())

        return {"x2": x2, "edge_index": ei}

    return cast(
        AllInput,
        {
            **__prepare_dnn_data(
                patents, dnn_categorical_fields, dnn_text_fields, batch_size
            ),
            **__prepare_gnn_input(patents, gnn_categorical_fields, batch_size),
        },
    )


# @classmethod
# def load_checkpoint(
#     cls, checkpoint_name: str, patents: Optional[Sequence[PatentApplication]] = None
# ):
#     """
#     Load model from checkpoint. If patents provided, will start training from the next epoch

#     Args:
#         patents (Sequence[PatentApplication]): List of patents
#         checkpoint_name (str): Checkpoint from which to resume
#     """
#     logging.info("Loading checkpoint %s", checkpoint_name)
#     model = CombinedModel(100, 100)  # TODO!!
#     checkpoint_file = os.path.join(CHECKPOINT_PATH, checkpoint_name)

#     if not os.path.exists(checkpoint_file):
#         raise Exception(f"Checkpoint {checkpoint_name} does not exist")

#     checkpoint = torch.load(checkpoint_file)
#     model.load_state_dict(checkpoint["model_state_dict"])
#     optimizer = OPTIMIZER_CLASS(model.parameters(), lr=LR)
#     optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

#     logging.info("Loaded checkpoint %s", checkpoint_name)

#     trainable_model = ModelTrainer(BATCH_SIZE, model, optimizer)

#     if patents:
#         trainable_model.train(patents, start_epoch=checkpoint["epoch"] + 1)

#     return trainable_model
