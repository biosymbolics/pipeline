"""
Binder NER model
"""
from typing import Iterable, Iterator, Union
from pydash import compact
import torch
from transformers import AutoTokenizer
import logging
from spacy.vocab import Vocab
from spacy.tokens import Doc

from .constants import NER_TYPES
from .types import Annotation
from .utils import extract_predictions, remove_overlapping_spans, prepare_features

logger = logging.getLogger(__name__)

DOC_STRIDE = 16
MAX_LENGTH = 128  # max??
DEFAULT_BASE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
DEFAULT_DEVICE = "mps"


class BinderNlp:
    """
    A class for running the Binder NLP model, partially emulating SpaCy's Language (nlp) class.

    To create the model, clone https://github.com/kristinlindquist/binder and from that directory
    ```
    $ python3
    config = {
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

    import torch, sys
    sys.path.append("src")
    torch.device('mps')
    from model import Binder
    from config import BinderConfig
    model = Binder(BinderConfig(**config))
    model.load_state_dict(torch.load('/tmp/pytorch_model.bin', map_location=torch.device('mps')))
    torch.save(model, 'model.pt')

    and copy model.pt into pipeline/
    ```
    """

    def __init__(self, model_file: str, base_model: str = DEFAULT_BASE_MODEL):
        device = torch.device(DEFAULT_DEVICE)
        self.model = torch.load(model_file)
        self.model.to(device)
        self.__tokenizer = AutoTokenizer.from_pretrained(base_model)

    def __call__(self, texts: list[str]):
        return self.pipe(texts)

    @property
    def type_map(self) -> dict[int, str]:
        """
        Get the type map for reconstituting the NER types
        """
        return dict([(i, t["name"]) for i, t in enumerate(NER_TYPES)])

    @property
    def __type_descriptions(self):
        """
        Get the type descriptions used by the Binder model
        """
        descriptions = self.tokenize(
            [t["description"] for t in NER_TYPES],
            {
                "padding": "longest",
                "return_tensors": "pt",
            },
        )
        return {
            "type_input_ids": descriptions["input_ids"],
            "type_attention_mask": descriptions["attention_mask"],
            "type_token_type_ids": descriptions["token_type_ids"],
        }

    @staticmethod
    def __get_doc(doc: Union[str, Doc], annotations: list[Annotation]) -> Doc:
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
        new_ents = [
            Doc.char_span(
                new_doc,
                a["start_char"],
                a["end_char"],
                label=a["entity_type"],
                alignment_mode="expand",
            )
            for a in annotations
        ]

        # re-create the existing ents, to reset ent start/end indexes
        existing_ents = [
            Doc.char_span(
                new_doc,
                e.start_char,
                e.end_char,
                label=e.label,
            )
            for e in (doc.ents if isinstance(doc, Doc) else [])
        ]

        all_ents = remove_overlapping_spans(compact(new_ents + existing_ents))

        new_doc.set_ents(all_ents)
        return new_doc

    def tokenize(
        self,
        text: Union[str, list[str]],
        tokenize_args: dict = {
            "return_overflowing_tokens": True,
            "return_offsets_mapping": True,
        },
    ):
        """
        Main tokenizer

        Args:
            text (Union[str, list[str]]): text or list of texts to tokenize
        """
        common_args = {
            "max_length": MAX_LENGTH,
            "stride": DOC_STRIDE,
            "return_tensors": "pt",
            "padding": "max_length",
            "truncation": True,
        }
        all_args = {**common_args, **tokenize_args}
        return self.__tokenizer(text, **all_args).to(DEFAULT_DEVICE)

    def extract(self, doc: Union[str, Doc]) -> Doc:
        """
        Extracts entity annotations for a given text.

        Args:
            doc (Union[str, Doc]): The Spacy Docs (or strings) to annotate.
        """
        text = doc.text if isinstance(doc, Doc) else doc
        features = prepare_features(text, self.tokenize(text))
        inputs = self.tokenize(text, {"return_tensors": "pt"})

        predictions = self.model(
            **inputs,
            **self.__type_descriptions,
        )

        annotations = extract_predictions(
            features, predictions.__dict__["span_scores"], self.type_map
        )
        return self.__get_doc(doc, annotations)

    def pipe(
        self,
        texts: Iterable[Union[str, Doc]],
    ) -> Iterator[Doc]:
        """
        Apply the pipeline to a batch of texts.
        Single threaded because GPU handles parallelism.

        Args:
            texts (Iterable[Union[str, Doc]]): The texts to annotate.
        """
        logging.info("Starting binder NER extraction")

        for text in texts:
            yield self.extract(text)
