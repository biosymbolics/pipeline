"""
Get info about R&D pipelines
"""
import logging
from datetime import date, datetime
from pydash import flatten
import polars as pl

from sources.sec.sec import fetch_quarterly_reports
from sources.sec.sec_client import extract_section


def get_pipeline(
    ticker: str, start_date: date, end_date: date = datetime.now()
) -> list[list[str]]:
    """
    Get the R&D pipeline for a given company
    TODO
    - sort by product
    - normalize names (ontology)
    - create date <-> product list
    """
    quarterly_reports = fetch_quarterly_reports(ticker, start_date, end_date)

    product_tables = flatten(
        map(
            lambda x: extract_section(x.get("linkToHtml")),
            quarterly_reports,
        )
    )

    # products = list(
    #     map(lambda table: table.select(pl.col(["product"])), product_tables)
    # )
    logging.info("Products: %s", product_tables)

    return product_tables
