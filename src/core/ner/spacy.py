"""
SpaCy client
"""

import logging
from typing import Any, Iterator
import spacy
from spacy.language import Language
from spacy.tokens import Doc
from thinc.api import set_gpu_allocator, prefer_gpu

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
            set_gpu_allocator("pytorch")
            prefer_gpu()

        self.model = model

        # disable additional_pipelines keys to we cana add them
        _exclude = [*exclude, *additional_pipelines.keys()]
        nlp: Language = spacy.load(self.model, exclude=_exclude, **kwargs)

        for name, args in additional_pipelines.items():
            nlp.add_pipe(name, **args)
            if name == "tok2vec" or name == "transformer":
                nlp.initialize()

        self._nlp = nlp

    def __getattr__(self, name):
        # Delegate attribute access to the underlying Language instance
        return getattr(self._nlp, name)

    def __call__(self, text: str) -> Any:
        return self._nlp(text)

    @classmethod
    def nlp(cls, text: str) -> Any:
        return cls.get_instance()._nlp(text)

    def pipe(self, *args, **kwargs) -> Iterator[Doc]:
        return self._nlp.pipe(*args, **kwargs)

    @classmethod
    def get_instance(cls, model: str = DEFAULT_MODEL, **kwargs) -> "Spacy":
        args = [("model", model), *sorted(kwargs.items())]
        args_hash = make_hashable(args)  # Convert args/kwargs to a hashable type
        if args_hash not in cls._instances:
            logger.debug("Returning UNCACHED nlp model (%s)", model)
            cls._instances[args_hash] = cls(model, **kwargs)
        logger.debug("Returning CACHED nlp model (%s)", model)
        return cls._instances[args_hash]


def get_transformer_nlp() -> Spacy:
    """
    Get a Spacy NLP model that uses a transformer
    """
    nlp = Spacy.get_instance(
        model="en_core_web_trf",
        # parser and tagger not currently "working" anyway. TODO: add back & make work!
        disable=["ner", "parser", "tagger"],
        additional_pipelines={
            "transformer": {
                "before": "tagger",
                "config": {
                    "model": {
                        "@architectures": "spacy-transformers.TransformerModel.v3",
                        "name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
                        "get_spans": {
                            "@span_getters": "spacy-transformers.strided_spans.v1",
                            "window": 128,
                            "stride": 96,
                        },
                        "tokenizer_config": {
                            "use_fast": True,
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
