"""
Get info about R&D pipelines
"""
from datetime import date, datetime
import json
import logging
from pydash import flatten
import polars as pl


from common.utils import ner
from common.utils.list import diff_lists
from common.clients.llama_index import create_and_query_index
from common.utils.validate import validate_or_pickle
from sources.sec.prompts import JSON_PIPELINE_PROMPT, JSON_PIPELINE_SCHEMA
from sources.sec.sec import fetch_annual_reports
from sources.sec import sec_client
from sources.sec.types import SecFiling
from sources.sec.types import SecProductQueryStrategy

logging.basicConfig(level="DEBUG")


def __df_to_products(df: pl.DataFrame) -> list[str]:
    """
    Get products from df and normalize
    """
    products = df.select(pl.col("product")).to_series().to_list()
    normalized = list(map(ner.normalize_entity_name, products))
    valid = list(filter(lambda p: p != "", normalized))
    return valid


def __search_for_products(sec_text: str, namespace: str, period: str) -> list[str]:
    """
    Uses LlamaIndex/GPT to extract product names
    """
    results = create_and_query_index(
        JSON_PIPELINE_PROMPT,
        namespace,
        period,
        [sec_text],
    )
    names = []
    # try:
    #     products = json.loads(results)
    #     for product in products:
    #         try:
    #             validate_or_pickle(product, JSON_PIPELINE_SCHEMA)
    #             names.append(product["brand_name"])
    #         except Exception as ex:
    #             logging.debug("Ignoring exception %s", ex)
    # except Exception as ex:
    #     print("WTF", ex)

    return names


def __extract_products(
    report: SecFiling, strategy: SecProductQueryStrategy = "TABLE"
) -> list[str]:
    """
    Extract R&D pipeline products from report
    """
    report_url = report.get("linkToHtml")
    logging.info(
        "Extracting normalized products via stategy %s (%s)", strategy, report_url
    )

    if strategy == "TABLE":
        product_tables = sec_client.extract_rd_pipeline(report_url)
        return flatten(map(__df_to_products, product_tables))

    if strategy == "SEARCH":
        section = sec_client.extract_section(report_url, return_type="text")
        return __search_for_products(
            namespace=report["ticker"],
            period=report.get("periodOfReport"),
            sec_text=section,
        )

    raise Exception("Strategy not recognized")


def extract_pipeline_by_period(
    reports: list[SecFiling], strategy: SecProductQueryStrategy = "TABLE"
) -> pl.DataFrame:
    """
    Extract R&D pipeline by reporting period
    """
    products_by_period = flatten(
        map(
            lambda report: {
                "period": report.get("periodOfReport"),
                "products": __extract_products(report, strategy),
            },
            reports,
        )
    )
    products_by_period = sorted(products_by_period, key=lambda r: r["period"])
    df = pl.DataFrame(products_by_period)
    return df


def diff_pipeline(products_by_period: pl.DataFrame) -> list[list[str]]:
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
    strategy: SecProductQueryStrategy = "TABLE",
) -> pl.DataFrame:
    """
    Get the R&D pipeline for a given company
    """
    logging.info("Grabbing annual reports for %s", ticker)
    annual_reports = fetch_annual_reports(ticker, start_date, end_date)

    logging.info("Extracting pipeline for %s", ticker)
    pipeline_df = extract_pipeline_by_period(annual_reports, strategy)

    logging.info("Grabbing diffs for %s", ticker)
    diffs = diff_pipeline(pipeline_df)
    pipeline_df = pipeline_df.with_columns(pl.Series(name="dropped", values=diffs))

    return pipeline_df
