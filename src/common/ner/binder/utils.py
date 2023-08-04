"""
Utility functions for the Binder NER model
"""

import re
import numpy as np
from pydash import compact, flatten
import torch
from transformers import BatchEncoding
from spacy.tokens import Span
import time
import logging
import polars as pl

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
    features: list[Feature], predictions: np.ndarray, type_map: dict[int, str]
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

        def create_annotation(tup: tuple[int, int, int], feature: Feature, idx: int):
            """
            Applies offset and creates an annotation.
            """
            offset_mapping = feature["offset_mapping"]
            start_char, end_char = (
                offset_mapping[tup[0]][0],  # type: ignore
                offset_mapping[tup[1]][1],  # type: ignore
            )
            pred = Annotation(
                id=f"{feature['id']}-{idx}",
                entity_type=type_map[tup[2]],
                start_char=int(start_char),
                end_char=int(end_char),
                text=feature["text"][start_char:end_char],
            )
            return pred

        start_time = time.time()
        cpu_span_logits = (
            span_logits.detach().cpu().clone().numpy()
        )  # https://github.com/pytorch/pytorch/issues/77764
        start_indexes, end_indexes, type_ids = start_end_types(cpu_span_logits, feature)
        logging.info(
            "Extracted predictions in %s seconds", round(time.time() - start_time, 2)
        )

        annotations = compact(
            [
                create_annotation(tup, feature, idx)
                for idx, tup in enumerate(zip(start_indexes, end_indexes, type_ids))
            ]
        )
        return annotations

    all_predictions = flatten(
        [__extract_prediction(predictions[0], feature) for feature in features]
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
    word_start_chars = [word[0] for word in word_idx]
    word_end_chars = [word[1] for word in word_idx]

    num_features = len(tokenized.pop("input_ids"))
    offset_mapping = tokenized.pop("offset_mapping")

    def process_feature(text: str, offset_mapping, i: int):
        feature: Feature = {
            "id": f"feat-{str(i + 1)}",
            "text": text,
            "token_start_mask": [],
            "token_end_mask": [],
            "offset_mapping": offset_mapping[i],
        }
        sequence_ids = tokenized.sequence_ids(i)

        # Create a DataFrame
        df = pl.DataFrame(
            {
                "offset_mapping": list(
                    offset_mapping[i].detach().cpu().clone().numpy()
                ),
                "sequence_ids": list(sequence_ids),
            }
        )

        # Split offset_mapping into two separate columns for start and end
        df = df.with_columns(
            (
                pl.col("offset_mapping").apply(lambda om: om[0] in word_start_chars)
                & ((pl.col("sequence_ids") == 0))
            ).alias("start_mask"),
            (
                pl.col("offset_mapping").apply(lambda om: om[1] in word_end_chars)
                & ((pl.col("sequence_ids") == 0))
            ).alias("end_mask"),
        )

        feature["token_start_mask"] = (
            df.select(pl.col("start_mask")).to_series().to_list()
        )
        feature["token_end_mask"] = df.select(pl.col("end_mask")).to_series().to_list()
        feature["offset_mapping"] = [
            o if sequence_ids[k] == 0 else None for k, o in enumerate(offset_mapping[i])
        ]
        return feature

    features = [process_feature(text, offset_mapping, i) for i in range(num_features)]
    return features


def has_span_overlap(new_ent: Span, index: int, existing_ents: list[Span]) -> bool:
    """
    Check if a new entity overlaps with any of the existing entities

    Args:
        new_ent: the new entity to check
        existing_ents: the existing entities to check against
    """
    has_overlap = any(
        new_ent.start_char <= ent.end_char and new_ent.end_char >= ent.start_char
        for ent in existing_ents[:index]
    )
    return has_overlap


def remove_overlapping_spans(spans: list[Span]) -> list[Span]:
    """
    Remove overlapping spans from a list of spans.
    If overlap, leave the longest.

    Args:
        spans: the spans to remove overlaps from
    """
    sorted_spans = sorted(spans, key=lambda e: e.end_char - e.start_char, reverse=True)
    return [
        span
        for idx, span in enumerate(sorted_spans)
        if not has_span_overlap(span, idx, sorted_spans)
    ]
