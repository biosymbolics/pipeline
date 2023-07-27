from git import Sequence
import numpy as np
from pydash import flatten
from typing import TypeGuard
import torch
from sklearn.calibration import LabelEncoder
import polars as pl
from clients.spacy import Spacy
from torch import nn
import torch.nn.functional as F

from typings.core import Primitive
from typings.patents import PatentApplication

from .constants import EMBEDDING_DIM, MAX_STRING_LEN


def is_tensor_list(
    embeddings: list[torch.Tensor] | list[Primitive],
) -> TypeGuard[list[torch.Tensor]]:
    return isinstance(embeddings[0], torch.Tensor)


# 5145 vs 1296
def get_input_dim(
    categorical_fields,
    text_fields=[],
    embedding_dim=EMBEDDING_DIM,
    text_embedding_dim=300,
):
    if len(text_fields) > 0:
        return 76920  # hack
    else:
        return 16560
    # input_length = (len(categorical_fields) * embedding_dim * 6) + (
    #     len(text_fields) * text_embedding_dim
    # )
    # return input_length


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
    """
    df = pl.from_dicts([patent for patent in patents])  # type: ignore
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

    label_encoders = [
        (field, encode_category(field, df)) for field in categorical_fields
    ]
    CatTuple = namedtuple("CatTuple", categorical_fields)  # type: ignore
    # all len 6
    cat_feats = [
        CatTuple(*fv)._asdict() for fv in zip(*[enc[1][1] for enc in label_encoders])
    ]

    nlp = Spacy.get_instance(disable=["ner"])

    cat_tensors: list[list[torch.Tensor]] = [
        [
            embedding_layers[field](torch.tensor(patent[field], dtype=torch.int))
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
    text_feats = [
        [
            torch.tensor(
                np.array(
                    [
                        token.vector
                        for value in get_string_values(patent, field)
                        for token in nlp(value)
                    ]
                ),
            )
            for field in text_fields
        ]
        for patent in patents
    ]
    # pad?? token_len x word_vec_len (300)

    def get_patent_features(i: int) -> torch.Tensor:
        combo_text_feats = (
            torch.flatten(torch.cat(text_feats[i])) if len(text_fields) > 0 else None
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
