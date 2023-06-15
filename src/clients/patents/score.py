from typing import TypedDict
import polars as pl
import math
import logging

from .constants import MAX_PATENT_LIFE

ExplainedScore = TypedDict("ExplainedScore", {"explanation": str, "score": float})
ScoreMap = dict[str, float]


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

    def __calc_score(attributes: list[str]) -> float:
        """
        Calculate the patent score based on its attributes.
        Uses softmax-like normalization.
        """
        score: float = sum([math.exp(score_map.get(attr, 0)) for attr in attributes])
        total_possible: float = sum(map(math.exp, score_map.values()))
        score = (
            score / total_possible if len(attributes) > 0 else 0.1
        )  # if no attributes, let's give it a small non-zero score

        return score

    def __score_and_explain(attributes: list[str]) -> ExplainedScore:
        """
        Calculate the patent score & generate explanation
        """
        score = __calc_score(attributes)
        explanations = [
            f"{attr}: {score_map.get(attr, 0)}"
            for attr in attributes
            if attr in score_map
        ]
        return {"score": score, "explanation": ", ".join(explanations)}

    scores = pl.col(attributes_column).apply(__score_and_explain)

    # score and explanation are in the same column, so we need to unnest
    df = df.with_columns(scores.alias("result")).unnest("result")

    # multiply score by pct patent life remaining
    df = df.with_columns(
        pl.col("score")
        .mul(df[years_column] / MAX_PATENT_LIFE)
        .alias("suitability_score"),
        pl.concat_list(pl.col(["score", "search_rank"])).alias("all_scores"),
    )

    # multiply score by search rank
    df = df.with_columns(
        pl.col("score").mul(df["search_rank"]).alias("search_score"),
    )

    return df


SUITABILITY_SCORE_MAP: ScoreMap = {
    "COMBINATION": 0,
    "COMPOUND_OR_MECHANISM": 2,
    "DIAGNOSTIC": -1.5,
    "FORMULATION": -0.5,
    "NOVEL": 1.5,
    "PREPARATION": -1,
    "PROCESS": -1,
    "METHOD": -1.5,
}


def calculate_score(df: pl.DataFrame) -> pl.DataFrame:
    """
    Calculate suitability score for a patent
    (Total/Possible) * (pct patent life remaining)

    Args:
        df (pl.DataFrame): The DataFrame with the patents data.
    """
    return score_patents(df, "attributes", "patent_years", SUITABILITY_SCORE_MAP)
