import random
from typing import Sequence, TypedDict
import polars as pl
import logging

from constants.patents import SUITABILITY_SCORE_MAP
from typings.companies import Company
from typings.patents import AvailabilityLikelihood, SuitabilityScoreMap

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
        >>> calc_suitability_score(['COMPOUND_OR_MECHANISM'], SUITABILITY_SCORE_MAP)
    """
    min = sum([v for v in score_map.values() if v <= 0])
    max = sum([v for v in score_map.values() if v >= 0])
    score: float = sum([(score_map.get(attr, 0)) for attr in attributes])

    if score < 0:
        return 0

    score = (score - min) / (max - min)
    return score


def calc_suitability_score_map(
    attributes_by_patent: Sequence[Sequence[str]], score_map: SuitabilityScoreMap
) -> list[ExplainedSuitabilityScore]:
    """
    Calculate the suitability score & generate explanation
    """

    def calc_score(attributes: Sequence[str]) -> ExplainedSuitabilityScore:
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

    return [calc_score(attrs) for attrs in attributes_by_patent]


def calc_adj_patent_years(py: int) -> int:
    """
    Calculate the adjusted patent years based on the patent years remaining.

    TODO:
    - patent term extensions (+5, no more than 14 years at approval - https://www.fr.com/insights/ip-law-essentials/intro-patent-term-extension/)
    - "new use" extension
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
    suit_score_df = pl.from_records(
        calc_suitability_score_map(
            df.select(pl.col(attributes_column)).to_series().to_list(),
            score_map=score_map,
        )  # type: ignore
    )

    # multiply score by pct patent life remaining
    df = (
        pl.concat([df, suit_score_df], how="horizontal")
        .with_columns(
            pl.col("patent_years")
            .apply(lambda py: calc_adj_patent_years(py))  # type: ignore
            .alias("adj_patent_years"),  # type: ignore
            pl.Series(
                name="probability_of_success",
                values=[random.betavariate(2, 8) for _ in range(len(df))],
            ),
            pl.Series(
                name="reformulation_score",
                values=[random.betavariate(2, 8) for _ in range(len(df))],
            ),
        )
        .with_columns(
            pl.col("suitability_score")
            .mul(df[years_column] / MAX_PATENT_LIFE)
            .add(pl.col("availability_score"))
            .alias("score"),
        )
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


def add_availability(df: pl.DataFrame, company_map: dict[str, Company]) -> pl.DataFrame:
    """
    Add availability likelihood to patents

    df must already have assignees, publication_number, and is_active columns
    """
    avail_likelihood_map: dict[str, tuple[AvailabilityLikelihood, str]] = {
        rec["publication_number"]: AvailabilityLikelihood.find_from_record(
            rec, company_map
        )
        for rec in df.to_dicts()
    }
    avail_explanation_map: dict[str, str] = {
        k: v[1] for k, v in avail_likelihood_map.items()
    }
    avail_cat_map = {k: v[0].value for k, v in avail_likelihood_map.items()}
    avail_score_map = {k: v[0].score for k, v in avail_likelihood_map.items()}

    return df.with_columns(
        pl.col("publication_number")
        .map_dict(avail_cat_map)
        .alias("availability_likelihood"),
        pl.col("publication_number")
        .map_dict(avail_explanation_map)
        .alias("availability_explanation"),
        pl.col("publication_number")
        .map_dict(avail_score_map)
        .alias("availability_score"),
    )
