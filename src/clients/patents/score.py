from typing import TypedDict
import polars as pl
import math
import logging
from constants.patents import SUITABILITY_SCORE_MAP

from typings.patents import SuitabilityScoreMap

from .constants import MAX_PATENT_LIFE

ExplainedScore = TypedDict("ExplainedScore", {"explanation": str, "score": float})


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def calc_patent_score(attributes: list[str], score_map: SuitabilityScoreMap) -> float:
    """
    Calculate the patent score based on its attributes.
    Uses softmax-like normalization.
    """
    # math.exp(0) == 1, so we need to add a score for missing attributes
    # (otherwise negative-scored patents will appear more suitable)
    missing_attr_score = len(score_map.keys()) - len(attributes)

    score: float = (
        sum([math.exp(score_map.get(attr, 0)) for attr in attributes])
        + missing_attr_score
    )
    total_possible: float = sum(map(math.exp, score_map.values()))

    # if no attributes, let's give it a small non-zero score
    score = (score / total_possible) if len(attributes) > 0 else 0.1

    return score


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

    def __score_and_explain(attributes: list[str]) -> ExplainedScore:
        """
        Calculate the patent score & generate explanation
        """
        upper_attributes = [attr.upper() for attr in attributes]
        score = calc_patent_score(upper_attributes, score_map)
        explanations = [
            f"{attr}: {score_map.get(attr, 0)}"
            for attr in upper_attributes
            if attr in score_map
        ]
        return {"score": score, "explanation": ", ".join(explanations)}

    # score and explanation are in the same column, so we need to unnest
    df = df.with_columns(
        pl.col(attributes_column).apply(__score_and_explain).alias("result")
    ).unnest("result")

    # multiply score by pct patent life remaining
    df = df.with_columns(
        pl.col("score").mul(df[years_column] / MAX_PATENT_LIFE).alias("score"),
    )

    return df


def calculate_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate suitability score for a patent
    (Total/Possible) * (pct patent life remaining)

    Args:
        df (pl.DataFrame): The DataFrame with the patents data.
    """
    return score_patents(df, "attributes", "patent_years", SUITABILITY_SCORE_MAP)
