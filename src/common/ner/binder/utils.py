"""
Utility functions for the Binder NER model
"""

import re
import numpy as np
from pydash import compact, flatten
import torch
from transformers import BatchEncoding
from spacy.tokens import Span

from .types import Feature, Annotation


def generate_word_indices(text: str) -> list[tuple[int, int]]:
    """
    Generate word indices for a text

    Args:
        text (str): text to generate word indices for
    """
    word_indices = []
    token_re = re.compile(r"[\s\n]")
    words = token_re.split(text)
    for idx, word in enumerate(words):
        start_char = sum([len(word) + 1 for word in words[:idx]])
        end_char = start_char + len(re.sub("[.,;]$", "", word))
        word_indices.append((start_char, end_char))
    return word_indices


def extract_predictions(
    features: list[Feature], predictions: np.ndarray
) -> list[Annotation]:
    """
    Extract predictions from a list of features.

    TODO: convuluted; factor. Also might be reasonable to return Spans instead of Annotations

    Args:
        features: the features from which to extract predictions.
        predictions: the span predictions from the model.
    """

    def __extract_prediction(span_logits, feature: Feature) -> list[Annotation]:
        """
        Extract predictions from a single feature.
        """

        def start_end_types(
            span_logits: torch.Tensor, feature: Feature
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
            """
            Extracts predictions from the tensor
            """
            token_start_mask = np.array(feature["token_start_mask"]).astype(bool)
            token_end_mask = np.array(feature["token_end_mask"]).astype(bool)

            # using the [CLS] logits as thresholds
            span_preds = np.triu(span_logits > span_logits[:, 0:1, 0:1])

            type_ids, start_indexes, end_indexes = (
                token_start_mask[np.newaxis, :, np.newaxis]
                & token_end_mask[np.newaxis, np.newaxis, :]
                & span_preds
            ).nonzero()

            return (start_indexes, end_indexes, type_ids)

        def create_annotation(tup: tuple[int, int, int], feature: Feature):
            """
            Applies offset and creates an annotation.
            """
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

    all_predictions = flatten(
        [
            __extract_prediction(predictions[idx], feature)
            for idx, feature in enumerate(features)
        ]
    )

    return all_predictions


def prepare_features(text: str, tokenized: BatchEncoding) -> list[Feature]:
    """
    Prepare features for torch model.

    Handles offsets.

    Args:
        text: the text to prepare features for.
        tokenized: the tokenized text.

    Note: this is kinda convuluted and should be refactored.
    """
    word_idx = generate_word_indices(text)
    word_start_chars = [[word[0] for word in word_idx]]
    word_end_chars = [[word[1] for word in word_idx]]

    num_features = len(tokenized.pop("input_ids"))
    sample_mapping = tokenized.pop("overflow_to_sample_mapping")
    offset_mapping = tokenized.pop("offset_mapping")

    def process_feature(text: str, offset_mapping, sample_mapping, i: int):
        def get_offset_chars(
            start_char: int,
            end_char: int,
            sequence_id: int | None,
            sample_index: int,
        ):
            if sequence_id != 0:
                return (0, 0)

            return (
                int(start_char in word_start_chars[sample_index]),
                int(end_char in word_end_chars[sample_index]),
            )

        feature: Feature = {
            "id": str(i + 1),
            "text": text,
            "token_start_mask": [],
            "token_end_mask": [],
            "offset_mapping": offset_mapping[i],
        }
        sequence_ids = tokenized.sequence_ids(i)

        token_masks = [
            get_offset_chars(
                om[0], om[1], sequence_ids[offset_index], sample_index=sample_mapping[i]
            )
            for offset_index, om in enumerate(feature["offset_mapping"])
            if om is not None
        ]
        feature["token_start_mask"] = [m[0] for m in token_masks]
        feature["token_end_mask"] = [m[1] for m in token_masks]
        feature["offset_mapping"] = [
            o if sequence_ids[k] == 0 else None for k, o in enumerate(offset_mapping[i])
        ]
        return feature

    features = [
        process_feature(text, offset_mapping, sample_mapping, i)
        for i in range(num_features)
    ]
    return features


def has_span_overlap(new_ent: Span, existing_ents: list[Span]) -> bool:
    """
    Check if a new entity overlaps with any of the existing entities
    """
    return not any(
        new_ent.start_char < ent.end_char and new_ent.end_char > ent.start_char
        for ent in existing_ents
    )
