"""
Patent client
"""
import time
from typing import Any, Sequence, TypedDict, cast
import polars as pl
import logging

from clients.patents.constants import DOMAINS_OF_INTEREST
from typings import PatentApplication
from utils.list import dedup

from .score import calculate_scores
from .utils import get_patent_years

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


def format_search_result(
    results: Sequence[dict[str, Any]]
) -> Sequence[PatentApplication]:
    """
    Format patent search results and adds scores

    Adds:
    - patent_years
    - *DOMAINS_OF_INTEREST

    Args:
        results (Sequence[PatentApplication]): patents search results & summaries
    """
    start = time.time()

    if len(results) == 0:
        raise ValueError("No results returned. Try adjusting parameters.")

    df = pl.from_dicts(
        results,
        infer_schema_length=None,  # slow
    )

    # group terms by domain (type)
    df = df.with_columns(
        get_patent_years("priority_date").alias("patent_years"),
        *[
            df.select(
                pl.struct(["terms", "domains"])
                .apply(lambda rec: filter_terms_by_domain(rec, d))  # type: ignore
                .alias(d)
            ).to_series()
            for d in DOMAINS_OF_INTEREST
        ],
    ).drop("terms", "domains")

    df = calculate_scores(df).sort("score").reverse()

    logging.info(
        "Took %s seconds to format %s results", round(time.time() - start, 2), len(df)
    )

    return cast(Sequence[PatentApplication], df.to_dicts())
