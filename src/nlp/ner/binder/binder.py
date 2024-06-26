"""
Binder NER model
"""

from typing import Iterable, Sequence
from pydash import compact
import torch
from transformers import AutoTokenizer
import logging
from spacy.tokens import Doc

from nlp.ner.spacy import get_transformer_nlp
from constants.core import (
    DEFAULT_BASE_TOKENIZER_MODEL,
    DEFAULT_TORCH_DEVICE,
    DEFAULT_NLP_MODEL_ARGS,
    DEFAULT_VECTORIZATION_MODEL,
)

from .constants import NER_TYPES
from .types import Annotation
from .utils import (
    extract_prediction,
    remove_overlapping_spans,
    prepare_feature,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BinderNlp:
    """
    A class for running the Binder NLP model, partially emulating SpaCy's Language (nlp) class.

    To create the model, clone https://github.com/kristinlindquist/binder and from that directory, run the instructions in the readme.
    """

    def __init__(self, model_file: str, base_model: str = DEFAULT_BASE_TOKENIZER_MODEL):
        device = torch.device(DEFAULT_TORCH_DEVICE)

        logger.info(
            "Loading torch model from: %s (device %s)", model_file, DEFAULT_TORCH_DEVICE
        )
        # self.model = nn.DataParallel(torch.load(model_file, map_location=device))
        self.model = torch.load(model_file, map_location=device)  # .to(device)

        self._tokenizer = AutoTokenizer.from_pretrained(
            base_model, use_fast=True, device_map="auto"
        )
        self.nlp = get_transformer_nlp(DEFAULT_VECTORIZATION_MODEL)

    def __call__(self, texts: list[str]):
        return self.pipe(texts)

    @property
    def type_map(self) -> dict[int, str]:
        """
        Get the type map for reconstituting the NER types
        """
        return dict([(i, t["name"]) for i, t in enumerate(NER_TYPES)])

    @property
    def type_descriptions(self):
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
    def add_entities_to_doc(doc: Doc, annotations: Sequence[Annotation]) -> Doc:
        """
        Add entity annotations to a SpaCy Doc.

        In the case of overlapping entity spans, takes the largest.

        Args:
            doc (str | Doc): text or SpaCy Doc
            annotations (Sequence[Annotation]): list of annotations
        """
        new_ents = [
            Doc.char_span(
                doc,
                a.start_char,
                a.end_char,
                label=a.entity_type,
            )
            for a in annotations
        ]

        if len(new_ents) < len(annotations):
            missed = [a for i, a in enumerate(annotations) if new_ents[i] is None]
            logger.warning(
                "Some entities were dropped creating Spacy Spans: %s", missed
            )

        # re-create the existing ents, to reset ent start/end indexes
        existing_ents = [
            Doc.char_span(
                doc,
                e.start_char,
                e.end_char,
                label=e.label,
            )
            for e in doc.ents
        ]

        all_ents = remove_overlapping_spans(compact(new_ents + existing_ents))
        doc.set_ents(all_ents)

        return doc

    def tokenize(
        self,
        text: str | list[str],
        tokenize_args: dict = {
            "return_overflowing_tokens": True,
            "return_offsets_mapping": True,
            "return_tensors": "pt",
        },
    ):
        """
        Main tokenizer

        Args:
            text (Union[str, list[str]]): text or list of texts to tokenize
        """
        all_args = {**DEFAULT_NLP_MODEL_ARGS, **tokenize_args}
        return self._tokenizer(text, **all_args).to(DEFAULT_TORCH_DEVICE)

    def extract(self, input: Doc | str) -> Doc:
        """
        Extracts entity annotations for a given text.

        Args:
            input (Doc): the string or SpaCy Doc to extract entities from

        ```
        from core.ner.binder.binder import BinderNlp
        b = BinderNlp("models/binder.pt")
        text="\n ".join([
        "Bioenhanced formulations comprising eprosartan in oral solid dosage form for the treatment of asthma, and hypertension."
        for i in range(3)
        ]) + " and some melanoma."
        b.extract(text).ents
        ```
        """
        if isinstance(input, str):
            text = input
        else:
            text = input.text

        tokenized = self.tokenize(text)
        feature = prepare_feature(text, 0, tokenized)

        tokenized.pop("overflow_to_sample_mapping")
        tokenized.pop("offset_mapping")

        outputs = self.model(**tokenized, **self.type_descriptions)

        annotations = extract_prediction(
            outputs.span_scores[0],
            feature,
            self.type_map,
        )

        if isinstance(input, str):
            doc = Doc(self.nlp.vocab(), words=tokenized.tokens())
        else:
            doc = input

        return self.add_entities_to_doc(doc, annotations)

    def pipe(
        self,
        texts: Iterable[str],
    ) -> Iterable[Doc]:
        """
        Apply the pipeline to a batch of texts.

        Args:
            texts (Iterable[str]): The texts to annotate.
        """
        logger.debug("Starting binder NER extraction")

        docs = self.nlp.pipe(texts)

        for doc in docs:
            yield self.extract(doc)
