from typing import Any
import spacy

from common.utils.args import make_hashable


class Spacy:
    """
    Wrapper for Spacy NLP model

    Instances are cached when get_instance is called; this is to avoid loading the model multiple times
    and the performance hit that comes with it.
    """

    _instances: dict[str, Any] = {}

    def __init__(
        self,
        # alt models: en_core_sci_scibert, en_ner_bionlp13cg_md, en_ner_bc5cdr_md
        model: str = "en_core_web_sm",
        **kwargs: Any,
    ):
        self.model = model
        self._nlp = spacy.load(self.model, **kwargs)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self._nlp(*args, **kwds)

    @classmethod
    def get_instance(cls, model: str = "en_core_web_sm", **kwargs):
        args = [("model", model), *sorted(kwargs.items())]
        args_hash = make_hashable(args)  # Convert args/kwargs to a hashable type
        if args_hash not in cls._instances:
            cls._instances[args_hash] = cls(model, **kwargs)
        return cls._instances[args_hash]

    @classmethod
    def nlp(cls, text: str) -> Any:
        return cls.get_instance()._nlp(text)
