"""
Parsing for tables containing products (SEC)
"""
import logging
import polars as pl
from bs4 import BeautifulSoup

from common.utils.html_parsing.html_table import get_table_headers, get_table_rows


PRODUCT_HEADER_STRINGS = ["product"]
INDICATION_HEADER_STRINGS = ["disease", "indication"]


def __is_product_header(header_text: str) -> bool:
    return any(filter(lambda str: str in header_text.lower(), PRODUCT_HEADER_STRINGS))


def __is_indication_header(header_text: str) -> bool:
    return any(
        filter(lambda str: str in header_text.lower(), INDICATION_HEADER_STRINGS)
    )


def __has_product_header(headers: list[str]) -> bool:
    return any(map(__is_product_header, headers))


def __is_product_table(table) -> bool:
    text_headers = get_table_headers(table)
    is_product_table = __has_product_header(text_headers)
    return is_product_table


def __normalize_header(text_header: str) -> str:
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


def __table_to_data_frame(table):
    text_headers = get_table_headers(table)
    contents = get_table_rows(table, text_headers)
    schema = __get_schema(text_headers)
    logging.debug("Table schema %s", schema)
    df = pl.DataFrame(contents, schema=schema).sort("product")
    return df


def parse_product_tables(html):
    """
    Parse out product tables
    """
    table_contents = []
    soup = BeautifulSoup(html, features="html.parser")
    tables = soup.find_all("table")
    product_tables = [table for table in tables if __is_product_table(table)]
    for table in product_tables:
        logging.info("Parsing table")
        try:
            df = __table_to_data_frame(table)
            table_contents.append(df)
            print(df)
        except Exception as ex:
            logging.error(ex)  # oops

    return table_contents
