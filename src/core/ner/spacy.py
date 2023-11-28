"""
SpaCy client
"""

import logging
from typing import Any
import spacy
from spacy.language import Language

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
        **kwargs: Any,
    ):
        """
        Initialize Spacy instance

        TODO: add disables!

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
        spacy.require_gpu()  # type: ignore

        self.model = model
        self._nlp: Language = spacy.load(self.model, **kwargs)
        for name, args in additional_pipelines.items():
            self._nlp.add_pipe(name, **args)

    def __getattr__(self, name):
        # Delegate attribute access to the underlying Language instance
        return getattr(self._nlp, name)

    def __call__(self, text: str) -> Any:
        return self._nlp(text)

    @classmethod
    def nlp(cls, text: str) -> Any:
        return cls.get_instance()._nlp(text)

    @classmethod
    def get_instance(cls, model: str = DEFAULT_MODEL, **kwargs) -> "Spacy":
        spacy.require_gpu()  # type: ignore
        args = [("model", model), *sorted(kwargs.items())]
        args_hash = make_hashable(args)  # Convert args/kwargs to a hashable type
        if args_hash not in cls._instances:
            logger.debug("Returning UNCACHED nlp model (%s)", model)
            cls._instances[args_hash] = cls(model, **kwargs)
        logger.debug("Returning CACHED nlp model (%s)", model)
        return cls._instances[args_hash]
