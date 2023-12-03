"""
Utility functions for the Binder NER model
"""

import numpy.typing as npt
from pydash import compact, flatten
import torch
from transformers import BatchEncoding
from spacy.tokens import Span
import logging

from .types import Feature, Annotation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def extract_prediction(
    span_logits, feature: Feature, type_map: dict[int, str]
) -> list[Annotation]:
    """
    Extract predictions from a single feature.
    """

    def start_end_types() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Extracts predictions from the tensor
        """
        # https://github.com/pytorch/pytorch/issues/77764
        token_start_mask = torch.tensor(feature["token_start_mask"], device="mps")
        token_end_mask = torch.tensor(feature["token_end_mask"], device="mps")
        # using the [CLS] logits as thresholds
        span_preds = torch.triu(span_logits > span_logits[:, 0:1, 0:1])
        type_ids, start_indexes, end_indexes = torch.bitwise_and(
            torch.bitwise_and(
                token_start_mask.unsqueeze(0).unsqueeze(2),
                token_end_mask.unsqueeze(0).unsqueeze(1),
            ),
            span_preds,
        ).nonzero()

        return start_indexes, end_indexes, type_ids

    def create_annotation(start, end, type, idx: int):
        """
        Applies offset and creates an annotation.
        """
        offset_mapping = feature["offset_mapping"]
        start_char, end_char = (
            offset_mapping[start][0],  # type: ignore
            offset_mapping[end][1],  # type: ignore
        )
        pred = Annotation(
            id=f"{feature['id']}-{idx}",
            entity_type=type_map[type],
            start_char=int(start_char),
            end_char=int(end_char),
            text=feature["text"][start_char:end_char],
        )
        return pred

    start_indexes, end_indexes, type_ids = start_end_types()

    annotations = compact(
        [
            create_annotation(*tup, idx)
            for idx, tup in enumerate(zip(start_indexes, end_indexes, type_ids))
        ]
    )
    return annotations


def extract_predictions(
    features: list[Feature],
    predictions: npt.NDArray,
    type_map: dict[int, str],
) -> list[Annotation]:
    """
    Extract predictions from a list of features.

    Args:
        features: the features from which to extract predictions.
        predictions: the span predictions from the model.
        type_map: the type map for reconstituting the NER types
    """
    all_predictions = flatten(
        [
            extract_prediction(predictions[i], feature, type_map)
            for i, feature in enumerate(features)
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
    """
    num_features = len(tokenized["input_ids"])  # type: ignore
    offset_mapping = tokenized.pop("offset_mapping")  # ugh mutation

    def process_feature(i: int):
        sequence_ids = tokenized.sequence_ids(i)

        offset_mapping_i = [(om[0].item(), om[1].item()) for om in offset_mapping[i]]
        feature: Feature = {
            "id": f"feat-{str(i + 1)}",
            "text": text,
            "token_start_mask": [om[0] for om in offset_mapping_i],
            "token_end_mask": [om[1] for om in offset_mapping_i],
            "offset_mapping": [
                om if sequence_ids[k] == 0 else None
                for k, om in enumerate(offset_mapping_i)
            ],
        }

        return feature

    features = [process_feature(i) for i in range(num_features)]
    return features


def has_span_overlap(new_ent: Span, index: int, existing_ents: list[Span]) -> bool:
    """
    Check if a new entity overlaps with any of the existing entities

    Args:
        new_ent: the new entity to check
        index: the index of the new entity
        existing_ents: the existing entities to check against
    """
    overlapping_ents = [
        ent
        for ent in existing_ents[:index]
        if new_ent.start_char <= ent.end_char and new_ent.end_char >= ent.start_char
    ]
    if len(overlapping_ents) > 0:
        logger.warning(
            "Overlap detected between %s and %s", new_ent.text, overlapping_ents
        )
        return True
    return False


def remove_overlapping_spans(spans: list[Span]) -> list[Span]:
    """
    Remove overlapping spans from a list of spans.
    If overlap, leave the longest.

    Args:
        spans: the spans from which to remove overlaps
    """
    sorted_spans = sorted(spans, key=lambda e: e.end_char - e.start_char, reverse=True)

    return [
        span
        for idx, span in enumerate(sorted_spans)
        if not has_span_overlap(span, idx, sorted_spans)
    ]
