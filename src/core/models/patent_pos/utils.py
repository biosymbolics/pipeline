"""
Utils for patent eNPV model
"""

import logging
import random
from git import Sequence
import numpy as np
from pydash import compact, flatten
from typing import TypeGuard, cast
import torch
from sklearn.calibration import LabelEncoder
import polars as pl
from clients.spacy import Spacy
from torch import nn
import torch.nn.functional as F
from collections import namedtuple
from sklearn.decomposition import PCA
from common.utils.list import batch, batch_as_tensors
from core.models.patent_pos.types import AllInput, DnnInput, GnnInput

from typings.core import Primitive
from typings.patents import ApprovedPatentApplication as PatentApplication

from .constants import EMBEDDING_DIM, MAX_STRING_LEN, MAX_CATS_PER_LIST, TEXT_FEATURES


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


def is_tensor_list(
    embeddings: list[torch.Tensor] | list[Primitive],
) -> TypeGuard[list[torch.Tensor]]:
    return isinstance(embeddings[0], torch.Tensor)


def get_string_values(patent: PatentApplication, field: str) -> list:
    val = patent.get(field)
    if isinstance(val, list):
        return [v[0:MAX_STRING_LEN] for v in val]
    if val is None:
        return [None]
    return [val[0:MAX_STRING_LEN]]


def encode_category(field, df):
    le = LabelEncoder()
    new_list = df.select(pl.col(field)).to_series().to_list()
    le.fit(flatten(new_list))
    values = [le.transform([x] if isinstance(x, str) else x) for x in new_list]
    return (le, values)


def get_feature_embeddings(
    patents: Sequence[PatentApplication],
    categorical_fields: list[str],
    text_fields: list[str] = [],
    embedding_dim: int = EMBEDDING_DIM,
):
    """
    Get embeddings for patent features

    Args:
        patents (Sequence[PatentApplication]): List of patents
        categorical_fields (list[str]): List of fields to embed as categorical variables
        text_fields (list[str]): List of fields to embed as text
        embedding_dim (int, optional): Embedding dimension. Defaults to EMBEDDING_DIM.
    """
    nlp = Spacy.get_instance(disable=["ner"])
    pca = PCA(n_components=TEXT_FEATURES)

    _patents = [*patents]
    random.shuffle(_patents)
    df = pl.from_dicts(_patents, infer_schema_length=1000)  # type: ignore
    size_map = dict(
        [
            (field, df.select(pl.col(field).flatten()).n_unique())
            for field in categorical_fields
        ]
    )
    embedding_layers = dict(
        [
            (field, nn.Embedding(size_map[field], embedding_dim))
            for field in categorical_fields
        ]
    )

    label_encoders = [encode_category(field, df) for field in categorical_fields]
    CatTuple = namedtuple("CatTuple", categorical_fields)  # type: ignore
    cat_feats = [
        CatTuple(*fv)._asdict() for fv in zip(*[enc[1] for enc in label_encoders])
    ]

    cat_tensors: list[list[torch.Tensor]] = [
        [
            embedding_layers[field](
                torch.tensor(patent[field][0:MAX_CATS_PER_LIST], dtype=torch.int)
            )
            for field in categorical_fields
        ]
        for patent in cat_feats
    ]
    max_len_0 = max(f.size(0) for f in flatten(cat_tensors))
    max_len_1 = max(f.size(1) for f in flatten(cat_tensors))
    padded_cat_tensors = [
        [
            F.pad(f, (0, max_len_1 - f.size(1), 0, max_len_0 - f.size(0)))
            if f.size(0) > 0
            else torch.empty([max_len_0, max_len_1])
            for f in cat_tensor
        ]
        for cat_tensor in cat_tensors
    ]
    cat_feats = [torch.cat(tensor_set) for tensor_set in padded_cat_tensors]

    if len(text_fields) > 0:
        text_vectors = [
            np.concatenate(
                [
                    np.array(
                        [
                            token.vector
                            for value in get_string_values(patent, field)
                            for token in nlp(value)
                        ]
                    )
                    for field in text_fields
                ]
            )
            for patent in patents
            if len(text_fields) > 0
        ]
        pca_model = pca.fit(np.concatenate(text_vectors, axis=0))
        text_feats = [
            torch.flatten(torch.tensor(pca_model.transform(text_vector)))
            for text_vector in text_vectors
        ]
        max_len = max(f.size(0) for f in flatten(text_feats))
        text_feats = [F.pad(f, (0, max_len - f.size(0))) for f in text_feats]

    def get_patent_features(i: int) -> torch.Tensor:
        combo_text_feats = (
            torch.flatten(text_feats[i]) if len(text_fields) > 0 else None
        )
        combo_cat_feats = torch.flatten(cat_feats[i])
        combined = (
            torch.cat([combo_cat_feats, combo_text_feats], dim=0)
            if combo_text_feats is not None
            else combo_cat_feats
        )
        return combined

    embeddings = [get_patent_features(i) for i, _ in enumerate(patents)]
    return embeddings


def __batch(
    items: list[torch.Tensor] | list[Primitive], batch_size: int
) -> torch.Tensor:
    """
    Batch a list of tensors or primitives

    Args:
        items (list[torch.Tensor] | list[Primitive]): List of tensors or primitives
        batch_size (int): Batch size
    """
    if not is_tensor_list(items):
        # then list of primitives
        batches = batch_as_tensors(cast(list[Primitive], items), batch_size)
        num_dims = 1
    else:
        num_dims = len(items[0].size()) + 1
        batches = batch(items, batch_size)
        batches = [torch.stack(b) for b in batches]

    def get_batch_pad(b: torch.Tensor):
        if num_dims == 1:
            return (0, batch_size - b.size(0))

        if num_dims == 2:
            return (0, 0, 0, batch_size - b.size(0))

        raise ValueError("Unsupported number of dimensions: %s" % num_dims)

    batches = [F.pad(b, get_batch_pad(b)) for b in batches]

    logging.info("Batches: %s (%s)", len(batches), [b.size() for b in batches])
    return torch.stack(batches)


def resize_and_batch(embeddings: list[torch.Tensor], batch_size: int) -> torch.Tensor:
    """
    Size embeddings into batches

    Args:
        embeddings (list[torch.Tensor]): List of embeddings
        batch_size (int): Batch size
    """
    max_len = max(e.size(0) for e in embeddings)
    padded_emb = [F.pad(f, (0, max_len - f.size(0))) for f in embeddings]
    return __batch(padded_emb, batch_size)


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
        embeddings = get_feature_embeddings(
            patents, dnn_categorical_fields, dnn_text_fields
        )
        x1 = resize_and_batch(embeddings, batch_size)

        y_vals = [
            (1.0 if patent["approval_date"] is not None else 0.0) for patent in patents
        ]
        y = __batch(cast(list[Primitive], y_vals), batch_size)
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
        embeddings = get_feature_embeddings(patents, gnn_categorical_fields)
        x2 = resize_and_batch(embeddings, batch_size)
        edge_index = [torch.tensor([i, i]) for i in range(len(patents))]
        ei = __batch(edge_index, batch_size)
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
