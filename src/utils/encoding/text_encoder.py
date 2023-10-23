from typing import Sequence
import logging
from pydash import compact
import numpy as np
import torch
import numpy.typing as npt
import polars as pl

from core.ner.spacy import Spacy
from utils.tensor import array_to_tensor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MAX_STRING_LEN = 200


class TextEncoder:
    def __init__(
        self,
        field: str | None = None,
        n_features: int = 5,
        device: str = "cpu",
    ):
        self.n_features = n_features
        self.field = field
        self.nlp = Spacy.get_instance(disable=["ner"])
        self.device = device

    def _encode_df(self, df: pl.DataFrame) -> torch.Tensor:
        if not self.field:
            raise ValueError("Cannot encode dataframe without field")

        values = df.select(pl.col(self.field)).to_series().to_list()
        return self._encode(values)

    def _encode(
        self,
        values: Sequence[str] | npt.NDArray,
    ) -> torch.Tensor:
        """
        Get text embeddings given a list of dict

        @see https://pytorch.org/docs/stable/generated/torch.pca_lowrank.html
        """

        def vectorize(value: str | list[str]) -> torch.Tensor:
            val = value if isinstance(value, str) else (", ").join(compact(value))
            vectors = [
                token.vector for token in self.nlp(val[0:DEFAULT_MAX_STRING_LEN])
            ]
            if len(vectors) == 0:
                return torch.Tensor()
            return torch.Tensor(np.concatenate(vectors))

        text_vectors = list(map(vectorize, values))
        max_size = max([f.size(0) for f in text_vectors])

        # A=Udiag(S)V^T
        # use cpu for this op: "The operator 'aten::linalg_qr.out' is not currently implemented for the MPS device."
        A = array_to_tensor(text_vectors, (len(text_vectors), max_size))
        U, S, V = torch.pca_lowrank(A)

        return torch.matmul(A, V[:, : self.n_features]).to(self.device)

    def fit_transform(
        self, data: pl.DataFrame | Sequence | npt.NDArray
    ) -> torch.Tensor:
        """
        Fit and transform a dataframe
        """
        if isinstance(data, pl.DataFrame):
            encoded_values = self._encode_df(data)
        else:
            encoded_values = self._encode(data)

        return encoded_values
