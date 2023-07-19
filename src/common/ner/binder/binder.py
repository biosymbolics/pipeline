"""
Binder NER model
"""
from typing import Iterable, Iterator, Union
import torch
from transformers import AutoTokenizer
import logging
from spacy.vocab import Vocab
from spacy.tokens import Doc, Span

from .constants import NER_TYPES
from .types import Annotation
from .utils import extract_predictions, has_span_overlap, prepare_features

logger = logging.getLogger(__name__)

DOC_STRIDE = 16
MAX_LENGTH = 128  # max??
DEFAULT_BASE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"


class BinderNlp:
    """
    A class for running the Binder NLP model, partially emulating SpaCy's Language (nlp) class.

    To create the model, clone https://github.com/kristinlindquist/binder and from that directory
    ```
    $ python3
    >>> config = {
        "cache_dir": "",
        "end_loss_weight": 0.2,
        "hidden_dropout_prob": 0.1,
        "init_temperature": 0.07,
        "linear_size": 128,
        "max_span_width": 129,
        "ner_loss_weight": 0.5,
        "pretrained_model_name_or_path": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract",
        "revision": "main",
        "span_loss_weight": 0.6,
        "start_loss_weight": 0.2,
        "threshold_loss_weight": 0.5,
        "use_auth_token": False,
        "use_span_width_embedding": True
    }

    >>> import torch
    >>> from model import Binder
    >>> from config import BinderConfig
    >>> model = Binder(BinderConfig(**config))
    >>> model.load_state_dict(torch.load('/tmp/pytorch_model.bin', map_location=torch.device('cpu')))
    >>> torch.save(model, 'model.pt')

    and copy model.pt into pipeline/
    ```
    """

    def __init__(self, model_file: str, base_model: str = DEFAULT_BASE_MODEL):
        self.model = torch.load(model_file)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer_args = {
            "max_length": MAX_LENGTH,
            "stride": DOC_STRIDE,
            "return_tensors": "pt",
            "padding": "max_length",
        }

    def __call__(self, texts: list[str]):
        inputs = self.tokenize(texts)
        outputs = self.model(inputs)

        return outputs

    @property
    def __type_descriptions(self):
        """
        Get the type descriptions used by the Binder model
        """
        descriptions = self.tokenizer(
            [t["description"] for t in NER_TYPES],
            **{
                "padding": "longest",
                "return_tensors": "pt",
                **self.tokenizer_args,
            },
        )
        return {
            "type_input_ids": descriptions["input_ids"],
            "type_attention_mask": descriptions["attention_mask"],
            "type_token_type_ids": descriptions["token_type_ids"],
        }

    def __get_doc(self, doc: Union[str, Doc], annotations: list[Annotation]) -> Doc:
        """
        Create a (pseudo) SpaCy Doc from a string and a list of annotations

        In the case of overlapping entity spans, takes the largest.

        Args:
            doc (Union[str, Doc]): text or SpaCy Doc
            annotations (list[Annotation]): list of annotations
        """
        new_doc = Doc(
            vocab=doc.vocab if isinstance(doc, Doc) else Vocab(),
            words=doc.text.split() if isinstance(doc, Doc) else doc.split(),
        )
        ents = [
            Doc.char_span(new_doc, a["start_char"], a["end_char"], label=a["text"])
            for a in annotations
        ]
        new_ents = [ent for ent in ents if not has_span_overlap(ent, ents)]
        new_doc.set_ents(new_ents)
        return new_doc

    def tokenize(self, texts: list[str]):
        return self.tokenizer(
            texts,
            **{
                "truncation": False,
                "return_overflowing_tokens": True,
                "return_offsets_mapping": True,
                **self.tokenizer_args,  # type: ignore
            },
        )

    def add_entities(self, doc: Union[str, Doc]) -> Doc:
        """
        Predicts entity annotations for a given text.

        Args:
            doc (Union[str, Doc]): The Spacy Docs (or strings) to annotate.
        """
        text = doc.text if isinstance(doc, Doc) else doc
        features = prepare_features(text, self.tokenize([text]))
        inputs = self.tokenizer(
            text,
            **{
                "return_tensors": "pt",
                **self.tokenizer_args,  # type: ignore
            },
        )

        predictions = self.model(
            **inputs,
            **self.__type_descriptions,
        ).__dict__

        annotations = extract_predictions(features, predictions["span_scores"])
        return self.__get_doc(doc, annotations)

    def pipe(
        self,
        texts: Iterable[Union[str, Doc]],
    ) -> Iterator[Doc]:
        """
        Apply the pipeline to a batch of texts.
        (currently only supports single-threaded processing)

        Args:
            texts (Iterable[Union[str, Doc]]): The texts to annotate.
        """
        docs = [self.add_entities(text) for text in texts]
        for doc in docs:
            yield doc