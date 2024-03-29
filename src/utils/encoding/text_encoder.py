from typing import Sequence, TypeAlias
import logging
import numpy as np
import torch
import numpy.typing as npt
import polars as pl

from nlp.ner.spacy import Spacy
from utils.tensor import array_to_tensor


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_MAX_STRING_LEN = 100
WORD_VECTOR_LENGTH = 200

TextEncoderData: TypeAlias = pl.DataFrame | Sequence | npt.NDArray


class TextEncoder:
    def __init__(self, field, max_items_per_cat: int):
        self.nlp = Spacy.get_instance(model="en_core_sci_lg", disable=["ner"])
        self.max_items_per_cat = max_items_per_cat
        self.field = field

    def _vectorize_values(self, values: Sequence[str]) -> tuple[int, torch.Tensor]:
        docs = list(self.nlp.pipe([v[0:DEFAULT_MAX_STRING_LEN] for v in values]))

        # max_items_per_cat x max_tokens x word_vector_length
        vectors = [
            torch.Tensor(np.array([t.vector for t in docs[i]]))
            for i in range(len(values))
        ][0 : self.max_items_per_cat]
        max_tokens = max([*map(len, vectors), 1])
        return max_tokens, array_to_tensor(
            vectors,
            (self.max_items_per_cat, max_tokens, WORD_VECTOR_LENGTH),
        )

    def vectorize(self, values: TextEncoderData) -> torch.Tensor:
        if isinstance(values, pl.DataFrame):
            values = values.select(pl.col(self.field)).to_series().to_list()

        vectored = list(map(self._vectorize_values, values))
        vectors = [v[1] for v in vectored]
        max_tokens = max([v[0] for v in vectored])

        return array_to_tensor(
            vectors,
            (len(values), self.max_items_per_cat, max_tokens, WORD_VECTOR_LENGTH),
        )

    def fit(self, values: torch.Tensor):
        # NO-OP
        logger.warning("FIT is a no-op for TextEncoder")
        return

    def fit_transform(self, values: TextEncoderData) -> torch.Tensor:
        text_vectors = self.vectorize(values)
        return text_vectors
