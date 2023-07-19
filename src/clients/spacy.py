"""
SpaCy client
"""

from typing import Any
import spacy

from common.utils.args import make_hashable


# small model lemmatizes antibodies as antibodies, large model as antibody
DEFAULT_MODEL = "en_core_web_lg"


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
        **kwargs: Any,
    ):
        self.model = model
        self._nlp = spacy.load(self.model, **kwargs)

    def __getattr__(self, name):
        # Delegate attribute access to the underlying Language instance
        return getattr(self._nlp, name)

    def __call__(self, text: str) -> Any:
        return self._nlp(text)

    @classmethod
    def nlp(cls, text: str) -> Any:
        return cls.get_instance()._nlp(text)

    @classmethod
    def get_instance(cls, model: str = DEFAULT_MODEL, **kwargs):
        args = [("model", model), *sorted(kwargs.items())]
        args_hash = make_hashable(args)  # Convert args/kwargs to a hashable type
        if args_hash not in cls._instances:
            cls._instances[args_hash] = cls(model, **kwargs)
        return cls._instances[args_hash]
