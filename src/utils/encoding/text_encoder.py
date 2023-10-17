from typing import Sequence, cast
import logging
from pydash import flatten
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import PCA

from core.ner.spacy import Spacy
from data.prediction.clindev.constants import InputRecord
from utils.tensor import array_to_tensor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MAX_STRING_LEN = 200


def get_string_values(item: dict, field: str) -> list:
    val = item.get(field)
    if isinstance(val, list):
        return [v[0:DEFAULT_MAX_STRING_LEN] for v in val]
    if val is None:
        return [None]
    return [val[0:DEFAULT_MAX_STRING_LEN]]


def get_text_embeddings(
    records: Sequence[InputRecord],
    text_fields: list[str],
    n_text_features: int,
    device: str = "cpu",
) -> torch.Tensor:
    """
    Get text embeddings given a list of dicts
    """
    nlp = Spacy.get_instance(disable=["ner"])
    pca = PCA(n_components=n_text_features)
    text_vectors = [
        np.concatenate(
            [
                tv
                for tv in [
                    np.array(
                        [
                            token.vector
                            for value in get_string_values(record._asdict(), field)
                            for token in nlp(value)
                        ]
                    )
                    for field in text_fields
                ]
                if len(tv.shape) > 1
            ]
        )
        for record in records
    ]

    pca_model = pca.fit(np.concatenate(text_vectors, axis=0))
    text_feats = [
        torch.flatten(torch.tensor(pca_model.transform(text_vector)))
        for text_vector in text_vectors
    ]
    max_len = max(f.size(0) for f in flatten(text_feats))
    text_feats = [F.pad(f, (0, max_len - f.size(0))) for f in text_feats]

    tensor = array_to_tensor(
        text_feats,
        (
            len(text_feats),
            len(text_feats[0]),
            *text_feats[0][0].shape,
        ),
    )

    return tensor.to(device)


# TODO
# class TextEncoder(Encoder):
#     def __init__(self, *args, **kargs):
#         super().__init__(LabelEncoder, *args, **kargs)
