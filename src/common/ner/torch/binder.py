from functools import partial
from typing import Any, Optional, TypedDict
import numpy as np
from pydash import compact, flatten
import torch
from transformers import AutoTokenizer
import logging

from scripts.patents.binder import generate_word_indices

from .constants import NER_TYPES

logger = logging.getLogger(__name__)

base_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(base_model)

DOC_STRIDE = 16
MAX_LENGTH = 128  # max??


class Annotation(TypedDict):
    id: str
    entity_type: int  # TODO: str
    start_char: int
    end_char: int
    text: str


Feature = TypedDict(
    "Feature",
    {
        "id": str,
        "text": str,
        "offset_mapping": list[Optional[tuple[int, int]]],
        "token_start_mask": list[int],
        "token_end_mask": list[int],
    },
)


def prepare_features(text: str) -> list[Feature]:
    """
    Prepare features for a single example.

    Args:
        text (str): text to prepare features for
    """
    word_idx = generate_word_indices(text)
    word_start_chars = [[word[0] for word in word_idx]]
    word_end_chars = [[word[1] for word in word_idx]]

    tokenized = tokenizer(
        [text],
        truncation=False,
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    def __get_offset_chars(
        start_char: int,
        end_char: int,
        index: int,
        sequence_ids: list[int | None],
        sample_index: int,
    ):
        if sequence_ids[index] != 0:
            return (0, 0)
        return (
            int(start_char in word_start_chars[sample_index]),
            int(end_char in word_end_chars[sample_index]),
        )

    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")
    num_features = len(tokenized.pop("input_ids"))

    def process_feature(i: int):
        feature: Feature = {
            "id": str(i + 1),
            "text": text,
            "token_start_mask": [],
            "token_end_mask": [],
            "offset_mapping": offset_mapping[i],
        }
        sequence_ids = tokenized.sequence_ids(i)

        get_offset_chars = partial(
            __get_offset_chars,
            sequence_ids=sequence_ids,
            sample_index=sample_mapping[i],
        )

        token_masks = [
            get_offset_chars(om[0], om[1], index)
            for index, om in enumerate(feature["offset_mapping"])
            if om is not None
        ]
        feature["token_start_mask"] = [m[0] for m in token_masks]
        feature["token_end_mask"] = [m[1] for m in token_masks]
        feature["offset_mapping"] = [
            o if sequence_ids[k] == 0 else None for k, o in enumerate(offset_mapping[i])
        ]
        return feature

    features = [process_feature(i) for i in range(num_features)]
    return features


def extract_prediction(span_logits, feature: Feature) -> list[Annotation]:
    """
    Extract predictions from a single feature.

    Args:
        span_logits: logits for all spans in the feature.
        feature: the feature from which to extract predictions.
    """

    def start_end_types(
        span_logits: torch.Tensor, feature: Feature
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        # masks for start and end indices.
        token_start_mask = np.array(feature["token_start_mask"]).astype(bool)
        token_end_mask = np.array(feature["token_end_mask"]).astype(bool)

        # We use the [CLS] logits as thresholds
        span_preds = np.triu(span_logits > span_logits[:, 0:1, 0:1])

        type_ids, start_indexes, end_indexes = (
            token_start_mask[np.newaxis, :, np.newaxis]
            & token_end_mask[np.newaxis, np.newaxis, :]
            & span_preds
        ).nonzero()

        return (start_indexes, end_indexes, type_ids)

    def create_annotation(tup: tuple[int, int, int], feature: Feature):
        offset_mapping = feature["offset_mapping"]
        start_char, end_char = (
            offset_mapping[tup[0]][0],  # type: ignore
            offset_mapping[tup[1]][1],  # type: ignore
        )
        pred = Annotation(
            id=feature["id"],
            entity_type=tup[2],
            start_char=start_char,
            end_char=end_char,
            text=feature["text"][start_char:end_char],
        )
        return pred

    start_indexes, end_indexes, type_ids = start_end_types(span_logits, feature)
    return compact(
        [
            create_annotation(tup, feature)
            for tup in zip(start_indexes, end_indexes, type_ids)
        ]
    )


def extract_predictions(
    features: list[Feature], predictions: np.ndarray
) -> list[Annotation]:
    """
    Extract predictions from a list of features.

    Args:
        features: the features from which to extract predictions.
        predictions: the span predictions from the model.
    """
    all_predictions = flatten(
        [
            extract_prediction(predictions[idx], feature)
            for idx, feature in enumerate(features)
        ]
    )

    return all_predictions


def predict(text: str) -> list[Annotation]:
    """
    Predicts annotations for a given text.

    Args:
        text (str): The text to annotate.
    """
    model = torch.load("model.pt")

    descriptions = tokenizer(
        [t["description"] for t in NER_TYPES],
        max_length=MAX_LENGTH,
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )

    features = prepare_features(text)
    inputs = tokenizer(
        text,
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        padding="max_length",
        return_tensors="pt",
    )

    predictions = model(
        **inputs,
        type_input_ids=descriptions["input_ids"],
        type_attention_mask=descriptions["attention_mask"],
        type_token_type_ids=descriptions["token_type_ids"],
    ).__dict__

    results = extract_predictions(features, predictions["span_scores"])
    return results
