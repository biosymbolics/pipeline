"""
Get info about R&D pipelines
"""
from datetime import date, datetime
import logging
from pydash import flatten
import polars as pl

from common.utils.list import diff_lists
from common.utils.llm.llama_index import create_and_query_index
from common.utils.ner import normalize_entity_name
from sources.sec.sec import fetch_quarterly_reports
from sources.sec.sec_client import extract_section
from sources.sec.types import SecFiling
from sources.sec.types import SecProductQueryStrategy

logging.basicConfig(level="DEBUG")


def __get_products_via_parse(df: pl.DataFrame) -> list[str]:
    """
    Get products from df and normalize
    """
    normalized = list(
        map(
            normalize_entity_name,
            df.select(pl.col("product")).to_series().to_list(),
        )
    )
    valid = list(filter(lambda p: p != "", normalized))
    return valid


def __get_products_via_llama(namespace: str, period: str, url: str) -> list[str]:
    return create_and_query_index(
        "what are the products in currently in development?", namespace, period, url
    )


def get_normalized_products(
    report: SecFiling, strategy: SecProductQueryStrategy = "TABLE_PARSE"
):
    """
    Get normalized products from report
    """
    logging.info("Getting normalized products via stategy %s", strategy)
    if strategy == "TABLE_PARSE":
        return flatten(
            map(
                __get_products_via_parse,
                extract_section(report.get("linkToHtml")),
            )
        )

    return __get_products_via_llama(
        namespace=report["ticker"],
        period=report.get("periodOfReport"),
        url=report.get("linkToHtml"),
    )


def extract_pipeline_by_period(
    reports: list[SecFiling], strategy: SecProductQueryStrategy = "TABLE_PARSE"
) -> pl.DataFrame:
    """
    Parse pipeline by period
    """
    products_by_period = flatten(
        map(
            lambda report: {
                "period": report.get("periodOfReport"),  # parse_date
                "products": get_normalized_products(report, strategy),
            },
            reports,
        )
    )
    products_by_period = sorted(products_by_period, key=lambda r: r["period"])
    df = pl.DataFrame(products_by_period)
    return df


def find_pipeline_diffs(products_by_period) -> list[list[str]]:
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
    ticker: str,
    start_date: date,
    end_date: date = datetime.now(),
    strategy: SecProductQueryStrategy = "TABLE_PARSE",
) -> pl.DataFrame:
    """
    Get the R&D pipeline for a given company
    TODO
    - sort by product
    - normalize names (ontology)
    """
    logging.info("Grabbing quarterly reports for %s", ticker)
    quarterly_reports = fetch_quarterly_reports(ticker, start_date, end_date)

    logging.info("Extracting pipeline for %s", ticker)
    pipeline_df = extract_pipeline_by_period(quarterly_reports, strategy)

    logging.info("Grabbing diffs for %s", ticker)
    diffs = find_pipeline_diffs(pipeline_df)
    pipeline_df = pipeline_df.with_columns(pl.Series(name="dropped", values=diffs))

    # pl.Config.set_tbl_rows(100)
    # logging.info("Products: %s", pipeline_df)

    return pipeline_df
