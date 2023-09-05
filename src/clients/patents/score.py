from functools import partial
import random
from typing import TypedDict
import polars as pl
import math
import logging
from constants.patents import SUITABILITY_SCORE_MAP

from typings.patents import SuitabilityScoreMap

from .constants import EST_MAX_CLINDEV, MAX_PATENT_LIFE

ExplainedSuitabilityScore = TypedDict(
    "ExplainedSuitabilityScore",
    {"suitability_score_explanation": str, "suitability_score": float},
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calc_suitability_score(
    attributes: list[str], score_map: SuitabilityScoreMap
) -> float:
    """
    Calculate the suitability score based on its attributes.
    (min/max)

    Example:
        >>> calc_suitability_score(['DEVICE', 'COMPOUND_OR_MECHANISM', 'METHOD', 'DIAGNOSTIC'], SUITABILITY_SCORE_MAP)
        >>> calc_suitability_score(['DEVICE', 'COMPOUND_OR_MECHANISM'], SUITABILITY_SCORE_MAP)
        >>> calc_suitability_score(['DEVICE'], SUITABILITY_SCORE_MAP)
    """
    min = sum([v for v in score_map.values() if v <= 0])
    max = sum([v for v in score_map.values() if v >= 0])
    score: float = sum([(score_map.get(attr, 0)) for attr in attributes])
    score = (score - min) / (max - min)
    return score


def calc_and_explain_suitability_score(
    attributes: list[str], score_map: SuitabilityScoreMap
) -> ExplainedSuitabilityScore:
    """
    Calculate the suitability score & generate explanation
    """
    upper_attributes = [attr.upper() for attr in attributes]
    score = calc_suitability_score(upper_attributes, score_map)
    explanations = [
        f"{attr}: {score_map.get(attr, 0)}"
        for attr in upper_attributes
        if attr in score_map
    ]
    return {
        "suitability_score": score,
        "suitability_score_explanation": ", ".join(explanations),
    }


def calc_adj_patent_years(py: int) -> int:
    """
    Calculate the adjusted patent years based on the patent years remaining.
    **FAKE**
    """
    variance_modifier = 10  # arbitrary-ish
    min_clindev = 6  # arbitrary
    # max clindev minus years transpired on patent
    # in real impl, will depend upon either 1) actual trials if available or 2) predictions
    avg_clindev = max(min_clindev, EST_MAX_CLINDEV - (MAX_PATENT_LIFE - py))
    return max(0, py - round(abs(random.gauss(avg_clindev, py / variance_modifier))))


def score_patents(
    df: pl.DataFrame,
    attributes_column: str,
    years_column: str,
    score_map: SuitabilityScoreMap,
) -> pl.DataFrame:
    """
    Score patents based on given attributes and score map.

    Args:
        df (pl.DataFrame): The DataFrame with the patents data.
        attributes_column (str): The column name containing the patent attributes.
        years_column (str): The column name containing the number of patent years left.
        score_map (Dict[str, int]): The map with the scores for each attribute.

    Returns:
        pl.DataFrame: The DataFrame with the patent scores and explanations.
    """

    # score and explanation are in the same column, so we need to unnest
    df = df.with_columns(
        pl.col(attributes_column)
        .apply(partial(calc_and_explain_suitability_score, score_map=score_map))
        .alias("result")
    ).unnest("result")

    # multiply score by pct patent life remaining
    df = df.with_columns(
        pl.col("patent_years")
        .apply(lambda py: calc_adj_patent_years(py))  # type: ignore
        .alias("adj_patent_years"),  # type: ignore
        pl.Series(
            name="availability_score",
            values=[random.betavariate(2, 6) for _ in range(len(df))],
        ),
        pl.Series(
            name="probability_of_success",
            values=[random.betavariate(2, 8) for _ in range(len(df))],
        ),
    ).with_columns(
        pl.col("suitability_score")
        .mul(df[years_column] / MAX_PATENT_LIFE)
        .add(pl.col("availability_score"))
        .add(pl.col("probability_of_success"))
        .mul(1 / 3)  # average
        .alias("score"),
    )

    return df


def calculate_scores(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate suitability score for a patent
    (Total/Possible) * (pct patent life remaining)

    Args:
        df (pl.DataFrame): The DataFrame with the patents data.
    """
    return score_patents(df, "attributes", "patent_years", SUITABILITY_SCORE_MAP)
