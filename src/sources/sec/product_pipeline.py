"""
Get info about R&D pipelines
"""
from datetime import date, datetime
import json
import logging
from pydash import flatten
import polars as pl


from common.utils.file import save_as_pickle
from common.utils.list import diff_lists
from common.utils.llm.llama_index import create_and_query_index
from common.utils.ner import normalize_entity_name
from common.utils.validate import validate_or_pickle
from sources.sec.prompts import JSON_PIPELINE_PROMPT, JSON_PIPELINE_SCHEMA
from sources.sec.sec import fetch_quarterly_reports
from sources.sec.sec_client import extract_product_pipeline, extract_section
from sources.sec.types import SecFiling
from sources.sec.types import SecProductQueryStrategy

logging.basicConfig(level="DEBUG")


def __parse_products(df: pl.DataFrame) -> list[str]:
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


def __search_for_products(namespace: str, period: str, url: str) -> list[str]:
    """
    Uses LlamaIndex/GPT to extract product names
    """
    sec_section = extract_section(url, "text")
    results = create_and_query_index(
        JSON_PIPELINE_PROMPT,
        namespace,
        period,
        [sec_section],
    )
    names = []
    try:
        save_as_pickle(results, "bigpickle.txt")
        products = json.loads(results)
        for product in products:
            try:
                validate_or_pickle(product, JSON_PIPELINE_SCHEMA)
                print(product)
                names.append(product["brand_name"])
            except Exception as ex:
                logging.debug("Ignoring exception %s", ex)
    except Exception as ex:
        print("WTF", ex)

    return names


def normalize_products(
    report: SecFiling, strategy: SecProductQueryStrategy = "TABLE_PARSE"
) -> list[str]:
    """
    Get normalized products from report
    """
    report_url = report.get("linkToHtml")
    logging.info(
        "Getting normalized products via stategy %s (%s)", strategy, report_url
    )

    if strategy == "TABLE_PARSE":
        return flatten(
            map(
                __parse_products,
                extract_product_pipeline(report_url),
            )
        )

    return __search_for_products(
        namespace=report["ticker"],
        period=report.get("periodOfReport"),
        url=report_url,
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
                "products": normalize_products(report, strategy),
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
    diffs = diff_pipeline(pipeline_df)
    pipeline_df = pipeline_df.with_columns(pl.Series(name="dropped", values=diffs))

    return pipeline_df
