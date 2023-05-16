"""
Utils for parsing html tables
"""
import logging
from pydash import compact


def __get_text(row_element) -> list[str]:
    return list(map(lambda e: e.text, row_element))


def __is_row_empty(cells: list[str]) -> bool:
    non_empty = len(compact(cells)) > 1
    return not non_empty


def __infer_table_header(rows: list[list[str]]) -> list[str]:
    for row in rows:
        if not __is_row_empty(row):
            return row
    return []


def __get_text_rows(table) -> list[list[str]]:
    trs = table.find_all("tr")
    cells = list(map(lambda tr: __get_text(tr.find_all("td")), trs))
    return cells


def get_table_headers(table) -> list[str]:
    """
    Parses table headers
    - either th if present
    - othrwise the first non-empty row
    """

    headers = table.find_all("th")

    if headers:
        return __get_text(headers)

    text_rows = __get_text_rows(table)
    return __infer_table_header(text_rows)


def __include_row(cells: list[str], headers: list[str]) -> bool:
    is_not_header = set(cells) != set(headers)
    return not __is_row_empty(cells) and is_not_header


def get_table_rows(table, headers: list[str] = []):
    """
    Parses table rows
    - excludes empties
    - excludes pseudo-header
    """
    text_rows = __get_text_rows(table)
    non_empty_rows = list(filter(lambda row: __include_row(row, headers), text_rows))
    return non_empty_rows
