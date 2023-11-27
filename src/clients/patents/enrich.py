"""
Patent client
"""
from functools import reduce
import time
from typing import Any, Sequence, TypedDict
import polars as pl
import logging

from typings import ScoredPatentApplication
from typings.companies import Company
from utils.list import dedup

from .constants import DOMAINS_OF_INTEREST
from .score import add_availability, calculate_scores
from .utils import get_patent_years, is_patent_stale

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TermDict = TypedDict("TermDict", {"terms": list[str], "domains": list[str]})


def filter_terms_by_domain(rec: TermDict, domain: str) -> list[str]:
    """
    Filter terms by domain
    (also dedups)
    """
    terms = [z[0] for z in list(zip(rec["terms"], rec["domains"])) if z[1] == domain]
    return dedup(terms)


def enrich_search_result(
    results: Sequence[dict[str, Any]], company_map: dict[str, Company]
) -> list[ScoredPatentApplication]:
    """
    Enrich patent with scores, patent years, etc.

    Adds:
    - patent_years
    - *DOMAINS_OF_INTEREST

    Args:
        results (list[PatentApplication]): patents search results & summaries
    """
    start = time.time()

    if len(results) == 0:
        raise ValueError("No results returned. Try adjusting parameters.")

    df = pl.from_dicts(
        results,
        # infrequently non-null, messing up type inference
        schema_overrides={"last_trial_update": pl.Date},
    )

    steps = [
        lambda _df: _df.with_columns(
            get_patent_years().alias("patent_years"),
            is_patent_stale().alias("is_stale"),
            pl.col("similar_patents")
            .apply(lambda l: [item for item in l if item.startswith("WO")])
            .alias("similar_patents"),
            *[
                df.select(
                    pl.struct(["terms", "domains"])
                    .apply(lambda rec: filter_terms_by_domain(rec, d))  # type: ignore
                    .alias(d)
                ).to_series()
                for d in DOMAINS_OF_INTEREST
            ],
        ).drop("terms", "domains"),
        lambda _df: calculate_scores(_df).sort("score").reverse(),
        lambda _df: add_availability(_df, company_map),
    ]

    enriched_df = reduce(lambda _df, step: step(_df), steps, df)

    logging.info(
        "Took %s seconds to format %s results", round(time.time() - start, 2), len(df)
    )

    return [ScoredPatentApplication(**p) for p in enriched_df.to_dicts()]
