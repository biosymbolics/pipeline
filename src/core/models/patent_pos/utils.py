from pydash import flatten
from typing import TypeGuard
import torch
from sklearn.calibration import LabelEncoder
import polars as pl

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
