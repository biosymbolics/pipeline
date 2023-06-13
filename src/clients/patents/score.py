from typing import TypedDict
import polars as pl
import math

from .constants import MAX_PATENT_LIFE

ScoreMap = dict[str, float]

ExplainedScore = TypedDict("ExplainedScore", {"score": float, "explanation": str})

SUITABILITY_SCORE_MAP: ScoreMap = {
    "COMBINATION": 0,
    "COMPOUND_OR_MECHANISM": 2,
    "DIAGNOSTIC": -1,
    "FORMULATION": -0.5,
    "NOVEL": 1.5,
    "PREPARATION": -1,
    "PROCESS": -1,
    "METHOD": -1.5,
}


def __calc_score(attributes: list[str], score_map: ScoreMap) -> float:
    """
    Calculate the patent score based on the provided attributes and score map.
    Apply a softmax-like function to each score to ensure it's greater than 0.

    Args:
        attributes (list[str]): The patent attributes.
        score_map (ScoreMap): The map with the scores for each attribute.

    Returns:
        tuple[float, str]: The patent score and explanation.
    """
    score: float = sum([score_map.get(attr, 0) for attr in attributes])
    total_possible: float = sum(map(math.exp, score_map.values()))
    score = math.exp(score) / total_possible

    return score


def __score_and_explain(attributes: list[str], score_map: ScoreMap) -> ExplainedScore:
    """
    Calculate the patent score based on the provided attributes and score map.
    Apply the exponential function to each score to ensure it's greater than 0.

    Args:
        attributes (list[str]): The patent attributes.
        score_map (ScoreMap): The map with the scores for each attribute.

    Returns:
        tuple[float, str]: The patent score and explanation.
    """
    # calc score
    score = __calc_score(attributes, score_map)

    # format simple score explanation
    explanations = [
        f"{attr}: {score_map.get(attr, 0)}" for attr in attributes if attr in score_map
    ]
    return {"score": score, "explanation": ", ".join(explanations)}


def score_patents(
    df: pl.DataFrame, attributes_column: str, years_column: str, score_map: ScoreMap
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
    scores = pl.col(attributes_column).apply(lambda a: __score_and_explain(a, score_map))  # type: ignore

    df = df.with_columns(scores.alias("result")).unnest("result")
    df = df.with_columns(
        pl.col("score").mul(df[years_column] / MAX_PATENT_LIFE).alias("score")
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
