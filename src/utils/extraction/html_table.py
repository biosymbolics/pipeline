"""
Utils for parsing html tables
TODO:
- handling of malformed tables like https://www.sec.gov/ix?doc=/Archives/edgar/data/14272/000001427222000196/bmy-20220930.htm ("Significant Product and Pipeline Approvals")
- handle of multiple-row column tables like https://www.sec.gov/ix?doc=/Archives/edgar/data/14272/000001427223000104/bmy-20230331.htm ("Product and Pipeline Developments")
"""
from bs4 import BeautifulSoup
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


def get_all_table_elements(html: str):
    """
    Gets all table elements and contents (including lonely rows and cells)

    Args:
        html (str): html to parse
    """
    soup = BeautifulSoup(html, features="html.parser")
    tables = soup.find_all("table")
    lonely_rows = soup.find_all("tr")
    lonely_cells = soup.find_all("cells")
    return tables + lonely_rows + lonely_cells
