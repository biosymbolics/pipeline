"""
Get info about R&D pipelines
"""
import logging
from datetime import date, datetime
from pydash import flatten
import polars as pl

from common.utils.date import parse_date
from common.utils.list import diff_lists
from common.utils.ner import normalize_entity_name
from sources.sec.sec import fetch_quarterly_reports
from sources.sec.sec_client import extract_section
from sources.sec.types import SecFiling


def __get_normalized_products(df: pl.DataFrame) -> list[str]:
    """
    Get products from df and normalize
    """
    return list(
        map(
            normalize_entity_name,
            df.select(pl.col("product")).to_series().to_list(),
        )
    )


def parse_pipeline_by_period(reports: list[SecFiling]) -> pl.DataFrame:
    """
    Parse pipeline by period
    """
    products_by_period = flatten(
        map(
            lambda report: {
                "period": report.get("periodOfReport"),
                "products": flatten(
                    map(
                        __get_normalized_products,
                        extract_section(report.get("linkToHtml")),
                    )
                ),
            },
            reports,
        )
    )
    products_by_period = sorted(products_by_period, key=lambda r: r["period"])
    df = pl.DataFrame(products_by_period)
    return df


def get_pipeline_diffs(products_by_period) -> list[list[str]]:
    """
    Get diffs of what was dropped from one period to the next
    - do this in polars?
    """
    records = products_by_period.to_dicts()
    diffs = []
    for idx, curr in enumerate(records):
        if idx == 0:
            diff = []  # keep count equal
        else:
            previous = records[idx - 1]
            if not previous["products"]:
                diff = []
            diff = diff_lists(previous["products"], curr["products"])

        diffs.append(diff)
    return diffs


def get_pipeline_by_ticker(
    ticker: str, start_date: date, end_date: date = datetime.now()
) -> pl.DataFrame:
    """
    Get the R&D pipeline for a given company
    TODO
    - sort by product
    - normalize names (ontology)
    """
    quarterly_reports = fetch_quarterly_reports(ticker, start_date, end_date)

    pipeline_df = parse_pipeline_by_period(quarterly_reports)
    diffs = get_pipeline_diffs(pipeline_df)
    pipeline_df = pipeline_df.with_columns(pl.Series(name="dropped", values=diffs))

    # pl.Config.set_tbl_rows(100)
    # logging.info("Products: %s", pipeline_df)

    return pipeline_df
