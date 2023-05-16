"""
Client for SEC API
"""
import logging

from sec_api import ExtractorApi, QueryApi, XbrlApi

from common.utils.html_parsing.product_table import parse_product_tables
from sources.sec.types import SecFiling

# logging.getLogger().setLevel(logging.INFO)

# SSM obv
API_KEY = "093ca4a8aaf5ce274e9651953d4fdf31b73d541c3db3a099adb2172ba39b1fce"


class SecApiClient:
    """
    Class for SEC API client
    """

    def __init__(self):
        self.client = None

    def get_client(self) -> QueryApi:
        """
        Returns client
        """
        query_api = QueryApi(api_key=API_KEY)
        return query_api

    def __call__(self):
        if not self.client:
            self.client = self.get_client()
        return self.client


sec_client = SecApiClient()


def __get_query(criteria: list[str], take: int = 100, skip: int = 0) -> str:
    """
    Gets SEC query given criteria
    """
    anded_criterial = " AND ".join(criteria)
    query = {
        "query": {
            "query_string": {
                "query": anded_criterial,
            }
        },
        "from": skip,
        "size": take,
        "sort": [{"filedAt": {"order": "desc"}}],
    }
    return query


def fetch_sec_docs(criteria: list[str]) -> list[SecFiling]:
    """
    Fetch SEC docs based on specified criteria
    e.g. ["ticker:PFE", "filedAt:{2020-01-01 TO 2020-12-31}", "formType:10-Q"] -> docs
    """
    query = __get_query(criteria)
    logging.info("Getting SEC docs with query %s", query)
    response = sec_client().get_filings(query)
    filings = response.get("filings")

    if not filings:
        raise KeyError("Response is missing 'filings'")

    return filings


def parse_xbrl(url: str):
    """
    Parse xbrl
    """
    xbrl_api = XbrlApi(API_KEY)
    xbrl_json = xbrl_api.xbrl_to_json(htm_url=url)
    return xbrl_json


def extract_section(url: str):
    """
    Extract section
    """
    extractor_api = ExtractorApi(API_KEY)
    section_html = extractor_api.get_section(url, "part1item2", "html")
    product_tables = parse_product_tables(section_html)

    return product_tables
