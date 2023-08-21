"""
Patent client
"""
from typing import Any, Sequence, TypedDict, cast
import polars as pl
import logging

from clients.patents.constants import DOMAINS_OF_INTEREST
from typings import PatentApplication

from .score import calculate_score
from .utils import get_patent_years

logger = logging.getLogger(__name__)

TermDict = TypedDict("TermDict", {"terms": list[str], "domains": list[str]})


def filter_terms_by_domain(rec: TermDict, domain: str) -> list[str]:
    """
    Filter terms by domain
    """
    terms = [z[0] for z in list(zip(rec["terms"], rec["domains"])) if z[1] == domain]
    return terms


def format_search_result(
    results: Sequence[dict[str, Any]]
) -> Sequence[PatentApplication]:
    """
    Format patent search results and adds scores

    Args:
        results (Sequence[PatentApplication]): patents search results & summaries
    """

    if len(results) == 0:
        raise ValueError("No results returned. Try adjusting parameters.")

    df = pl.from_dicts(results)

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

    df = calculate_score(df).sort("search_score").reverse()

    return cast(Sequence[PatentApplication], df.to_dicts())
