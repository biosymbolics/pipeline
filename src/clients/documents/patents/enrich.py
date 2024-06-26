"""
Patent client
"""

from functools import reduce
import time
from typing import Sequence, TypedDict
import polars as pl
import logging
from prisma.partials import PatentDto

from typings import ScoredPatent
from typings.companies import CompanyFinancials
from utils.list import dedup

from .score import availability_exprs, calculate_scores
from .utils import filter_similar_patents, get_patent_years

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
    results: Sequence[PatentDto], financial_map: dict[str, CompanyFinancials]
) -> list[ScoredPatent]:
    """
    Enrich patent with scores, patent years, etc.

    - Adds patent_years, availability info and scores
    - Filters similar_patents

    Args:
        results (list[Patent]): patents search results & summaries
    """
    start = time.time()

    if len(results) == 0:
        logger.info("No results to enrich")
        return []

    pl.Config.activate_decimals()  # otherwise butchers decimal values
    df = pl.from_dicts(results, infer_schema_length=None)  # type: ignore

    steps = [
        lambda _df: _df.with_columns(
            get_patent_years().alias("patent_years"),
            filter_similar_patents().alias("similar_patents"),
            *availability_exprs(_df, financial_map),
        ),
        lambda _df: calculate_scores(_df).sort("score").reverse(),
    ]

    enriched_df = reduce(lambda _df, step: step(_df), steps, df)

    logging.info(
        "Took %s seconds to format %s results", round(time.time() - start, 2), len(df)
    )

    return [ScoredPatent(**p) for p in enriched_df.to_dicts()]
