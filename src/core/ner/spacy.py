"""
SpaCy client
"""

from collections.abc import Iterable
import logging
from typing import Any, Iterator
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from spacy.vocab import Vocab
from thinc.api import set_gpu_allocator
import torch

from constants.core import DEFAULT_VECTORIZATION_MODEL, DEFAULT_TORCH_DEVICE
from utils.args import make_hashable

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# sm & med model lemmatizes antibodies as `antibodie`, large model as `antibody`
DEFAULT_MODEL: str = "en_core_web_lg"


class Spacy:
    """
    Wrapper for Spacy NLP model

    Instances are cached when get_instance is called; this is to avoid loading the model multiple times
    and the performance hit that comes with it.
    """

    _instances: dict[str, Any] = {}

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        additional_pipelines: dict[str, dict] = {},
        exclude: list[str] = [],
        **kwargs: Any,
    ):
        """
        Initialize Spacy instance

        Using additions:
        rule_nlp = Spacy.get_instance(
            model="en_core_sci_lg",
            additional_pipelines={
                "merge_entities": {"after": "ner"},
                "entity_ruler": {
                    "config": {"validate": True, "overwrite_ents": True},
                    "after": "merge_entities",
                },
            },
        )
        """
        # acceleration via https://github.com/explosion/thinc-apple-ops
        # details: https://github.com/explosion/spaCy/discussions/12713
        spacy.prefer_gpu()  # type: ignore

        if model.endswith("_trf"):
            logger.warning("Setting GPU allocator to pytorch")
            set_gpu_allocator("pytorch")

        self.model = model

        # disable additional_pipelines keys to we cana add them
        _exclude = [*exclude, *additional_pipelines.keys()]
        self._nlp: Language = spacy.load(self.model, exclude=_exclude, **kwargs)

        # hack to deal with memory pigginess of spacy vocab
        # https://github.com/explosion/spaCy/discussions/9369
        self.inital_model = self._nlp.to_bytes()
        self.initial_vocab = set(self._nlp.vocab.strings)

        for name, args in additional_pipelines.items():
            self._nlp.add_pipe(name, **args)
            if name == "transformer":
                self._nlp.get_pipe(name).initialize(lambda: iter([]))

    def __getattr__(self, name):
        # Delegate attribute access to the underlying Language instance
        return getattr(self._nlp, name)

    def vocab(self) -> Vocab:
        return self._nlp.vocab

    def reset(self) -> None:
        """
        Reset the underlying nlp model
        """
        # reload from copy
        self._nlp.from_bytes(self.inital_model)

        # reset vocab
        self._nlp.vocab.strings._reset_and_load(self.initial_vocab)

    def __call__(self, text: str) -> Doc:
        return self._nlp(text)

    @classmethod
    def nlp(cls, text: str) -> Any:
        return cls.get_instance()._nlp(text)

    def pipe(self, *args, **kwargs) -> Iterator[Doc]:
        return self._nlp.pipe(*args, **kwargs)

    def load(self, *args, **kwargs) -> Any:
        return self.get_instance(*args, **kwargs)

    @classmethod
    def get_instance(cls, model: str = DEFAULT_MODEL, **kwargs) -> "Spacy":
        args = [("model", model), *sorted(kwargs.items())]
        args_hash = make_hashable(args)  # Convert args/kwargs to a hashable type
        if args_hash not in cls._instances:
            logger.info("Returning UNCACHED nlp model (%s)", model)
            cls._instances[args_hash] = cls(model, **kwargs)
        logger.debug(
            "Returning nlp model (%s) (all models: %s)", model, cls._instances.keys()
        )
        return cls._instances[args_hash]


def get_transformer_nlp(model: str = DEFAULT_VECTORIZATION_MODEL) -> Spacy:
    """
    Get a Spacy NLP model that uses a transformer
    """
    torch.set_num_threads(1)

    nlp = Spacy.get_instance(
        model="en_core_web_trf",
        exclude=[
            "ner",
            "attribute_ruler",
            "parser",
            "tagger",
            "lemmatizer",
        ],
        additional_pipelines={
            "transformer": {
                "config": {
                    "model": {
                        "@architectures": "spacy-transformers.TransformerModel.v3",
                        "name": model,
                        "get_spans": {
                            "@span_getters": "spacy-transformers.strided_spans.v1",
                            "window": 128,
                            "stride": 96,
                        },
                        "tokenizer_config": {
                            "use_fast": True,
                            "model_max_length": 512,
                            "device": DEFAULT_TORCH_DEVICE,
                            "mixed_precision": True,
                        },
                    },
                },
            },
            "tok2vec": {
                "config": {
                    "model": {
                        "@architectures": "spacy-transformers.TransformerListener.v1",
                        "pooling": {"@layers": "reduce_mean.v1"},
                    }
                },
            },
        },
    )

    return nlp
