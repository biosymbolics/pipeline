"""
Utility functions for the Binder NER model
"""

import torch
from transformers import BatchEncoding
from spacy.tokens import Span
import polars as pl
import logging

from nlp.ner.binder.types import Feature, Annotation

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SPAN_LENGTH = 15


def extract_prediction(
    span_logits: torch.Tensor,
    feature: Feature,
    type_map: dict[int, str],
) -> list[Annotation]:
    """
    Extracts predictions from the tensor
    """
    if feature["offset_mapping"] is None:
        return []

    start_offset_mapping = {
        i: (tup[0] if tup is not None else None)
        for i, tup in enumerate(feature["offset_mapping"])
    }
    end_offset_mapping = {
        i: (tup[1] if tup is not None else None)
        for i, tup in enumerate(feature["offset_mapping"])
    }

    # using the [CLS] logits as thresholds
    span_preds = torch.triu(span_logits > span_logits[:, 0:1, 0:1])

    # get mask of all valid spans
    mask = (
        feature["token_start_mask"].unsqueeze(0).unsqueeze(2)
        & feature["token_end_mask"].unsqueeze(0).unsqueeze(1)
        & span_preds
    )

    scores = span_logits[mask].unsqueeze(1)

    # slow - mask.nonzero() can take 1-2 seconds
    scores = torch.concat([mask.nonzero(), scores], dim=1)

    if scores.size(0) == 0:
        logger.warning("No valid spans found")
        return []

    df = (
        pl.from_numpy(
            scores.cpu().detach().numpy(),
            schema={
                "type": pl.Int8,
                "start": pl.Int16,
                "end": pl.Int16,
                "score": pl.Float32,
            },
        )
        .with_columns(
            pl.col("start").replace(start_offset_mapping).alias("start_char"),
            pl.col("end").replace(end_offset_mapping).alias("end_char"),
            pl.col("type").replace(type_map).alias("entity_type"),
        )
        .filter(
            ((pl.col("end") - pl.col("start")) < MAX_SPAN_LENGTH)
            & (pl.col("start_char") is not None)
            & (pl.col("end_char") is not None)
        )
        .with_columns(
            pl.struct(["start_char", "end_char"])
            .map_elements(
                lambda r: feature["text"][r["start_char"] : r["end_char"]],
                return_dtype=pl.Utf8,
            )
            .alias("text"),  # type: ignore
        )
        .sort(by="score", descending=True)
        .unique(("start", "end"), maintain_order=True)
    )

    return [
        Annotation(**r, id=f"{feature['id']}-{i}")
        for i, r in enumerate(df.drop(["type", "start", "end", "score"]).to_dicts())
    ]


def prepare_feature(text: str, index: int, tokenized: BatchEncoding) -> Feature:
    """
    Prepare features for torch model.

    Handles offsets.

    Args:
        text: the text to prepare features for.
        tokenized: the tokenized text.
    """
    offset_mapping = tokenized.offset_mapping[index]

    feature: Feature = {
        "id": f"feat-{str(index + 1)}",
        "text": text,
        "token_start_mask": offset_mapping[:, 0].bool(),
        "token_end_mask": offset_mapping[:, 1].bool(),
        "offset_mapping": [
            (om if i == 1 or om != (0, 0) else None)
            for i, om in enumerate(offset_mapping)
        ],
    }
    return feature


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
        logger.debug(
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
