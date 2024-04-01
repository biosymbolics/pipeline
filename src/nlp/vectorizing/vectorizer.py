import gc
from sentence_transformers import SentenceTransformer
import torch
import logging

from constants.core import DEFAULT_DEVICE, DEFAULT_VECTORIZATION_MODEL
from utils.args import make_hashable

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Vectorizer:
    """
    Vectorize text using a transformer model
    """

    _instances: dict[str, "Vectorizer"] = {}

    def __init__(
        self, model: str = DEFAULT_VECTORIZATION_MODEL, device: str = DEFAULT_DEVICE
    ):
        self.model = model
        self.device = device
        self.embedding_model = SentenceTransformer(model, device=device)

    def vectorize_string(self, text: str) -> torch.Tensor:
        return self.vectorize([text])[0]

    def vectorize(self, texts: list[str]) -> list[torch.Tensor]:
        if self.embedding_model is None:
            raise ValueError("Embedding model is not initialized")

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

    def empty_cache(self):
        """
        Consider calling periodically to free up memory
        (doesn't seem like this should be necessary... possibly a MPS-specific bug?)
        """
        self.embedding_model = None
        gc.collect()
        torch.mps.empty_cache()
        self.embedding_model = SentenceTransformer(self.model, device=self.device)

    def __call__(self, text: str) -> torch.Tensor:
        return self.vectorize_string(text)

    @classmethod
    def get_instance(
        cls, model: str = DEFAULT_VECTORIZATION_MODEL, **kwargs
    ) -> "Vectorizer":
        args = [("model", model), *sorted(kwargs.items())]
        args_hash = make_hashable(args)  # Convert args/kwargs to a hashable type
        if args_hash not in cls._instances:
            logger.info("Returning UNCACHED vectorizer (%s)", model)
            cls._instances[args_hash] = cls(model, **kwargs)
        logger.debug(
            "Returning vectorizer (%s) (all models: %s)", model, cls._instances.keys()
        )
        return cls._instances[args_hash]
