"""
Report utils
"""

import logging
from typing import Sequence
import polars as pl


from .types import CartesianDimension, DocumentReportRecord


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def apply_dimension_limit(
    data: Sequence[DocumentReportRecord], limit_field: CartesianDimension, limit: int
):
    """
    Apply a limit based on the counts on one of the dimensions

    TODO: generalize, and optimize perf.

    Args:
        data (Sequence[DocumentReportRecord]): report data
        limit_field (Literal["x", "y"]): field to limit upon.
                e.g. if x, limit total number of distinct x values but ensure all corresponding ys remain
        limit (int): limit
    """
    grouped = (
        pl.DataFrame(data)
        .groupby(limit_field)
        .agg(pl.sum("count").alias("count"))
        .sort("count", descending=True)
    )
    top_n = grouped["x"].limit(limit).to_list()

    return [d for d in data if d[limit_field] in top_n]
