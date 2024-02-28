from sentence_transformers import SentenceTransformer
import torch

from constants.core import DEFAULT_VECTORIZATION_MODEL
from data.prediction.constants import DEFAULT_DEVICE


class Vectorizer:
    """
    Vectorize text using a transformer model
    """

    def __init__(
        self, model: str = DEFAULT_VECTORIZATION_MODEL, device: str = DEFAULT_DEVICE
    ):
        self.embedding_model = SentenceTransformer(model, device="mps")

    def vectorize_string(self, text: str) -> torch.Tensor:
        return self.vectorize([text])[0]

    def vectorize(self, texts: list[str]) -> list[torch.Tensor]:
        with torch.no_grad():
            tensors = self.embedding_model.encode(texts)

        if isinstance(tensors, torch.Tensor):
            return [tensors]
        if (
            isinstance(tensors, list)
            and len(tensors) > 0
            and isinstance(tensors[0], torch.Tensor)
        ):
            return tensors

        return [torch.tensor(t) for t in tensors]

    def __call__(self, text: str) -> torch.Tensor:
        return self.vectorize_string(text)
