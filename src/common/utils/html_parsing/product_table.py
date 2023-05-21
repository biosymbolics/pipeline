"""
Parsing for tables containing products (SEC)
"""
import logging
from typing import Literal
import polars as pl
from bs4 import BeautifulSoup

from common.utils.html_parsing.html_table import get_table_headers, get_table_rows


PRODUCT_HEADER_STRINGS: list[str] = ["product"]
INDICATION_HEADER_STRINGS: list[str] = ["disease", "indication"]


def __is_product_header(header_text: str) -> bool:
    """
    Is this specific header for "products"?
    """

    def __in_header(s):
        return s in header_text.lower()

    return any(filter(__in_header, PRODUCT_HEADER_STRINGS))


def __is_indication_header(header_text: str) -> bool:
    """
    Is this specific header for "indication"?
    """

    def __in_header(s):
        return s in header_text.lower()

    return any(filter(__in_header, INDICATION_HEADER_STRINGS))


def __has_product_header(headers: list[str]) -> bool:
    """
    Does this table appear to have a header column for products?
    """
    return any(map(__is_product_header, headers))


def __is_product_table(table) -> bool:
    """
    Does this table appear to be a product table?
    """
    text_headers = get_table_headers(table)
    is_product_table = __has_product_header(text_headers)
    return is_product_table


def __normalize_header(text_header: str) -> str:
    """
    Normalizes the name of headers in a product table
    """
    if __is_product_header(text_header):
        return "product"
    if __is_indication_header(text_header):
        return "indication"
    return text_header


def __get_schema(text_headers: list[str]) -> dict:
    """
    Creates a schema with normalized field/column names
    """
    headers = [header for header in text_headers if header != ""]
    schema = dict((__normalize_header(header), str) for header in headers)
    return schema


def __table_to_data_frame(table) -> pl.DataFrame:
    """
    Turns a product table into a dataframe
    """
    text_headers = get_table_headers(table)
    contents = get_table_rows(table, text_headers)
    schema = __get_schema(text_headers)
    logging.debug("Table schema %s", schema)
    df = pl.DataFrame(contents, schema=schema).sort("product")
    return df


def extract_product_tables(html: str) -> list[pl.DataFrame]:
    """
    Extract product tables from SEC docs
    """
    table_dfs = []
    soup = BeautifulSoup(html, features="html.parser")
    tables = soup.find_all("table")
    product_tables = list(filter(__is_product_table, tables))

    for table in product_tables:
        logging.info("Parsing product table")
        try:
            df = __table_to_data_frame(table)
            table_dfs.append(df)
        except Exception as ex:
            logging.warning("Error parsing table: %s", ex)  # oops

    return table_dfs
