"""
Client for SEC API
"""

import logging
import os
from sec_api import QueryApi

from .types import SecFiling


API_KEY = os.environ["SEC_API_KEY"]


class SecClient:
    """
    Class for SEC client
    """

    def __init__(self):
        self.client = QueryApi(api_key=API_KEY)

    def _get_query(self, criteria: list[str], take: int = 100, skip: int = 0) -> dict:
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

    def fetch_docs(self, criteria: list[str]) -> list[SecFiling]:
        """
        Fetch SEC docs based on specified criteria
        e.g. ["ticker:PFE", "filedAt:{2020-01-01 TO 2020-12-31}", "formType:10-K"] -> docs
        """
        query = self._get_query(criteria)
        logging.info("Getting SEC docs with query %s", query)
        response = self.client.get_filings(query)

        if not response:
            raise Exception("No response from SEC API")

        filings = response.get("filings")

        if not filings:
            logging.error("Response is missing 'filings': %s", response)
            raise KeyError("Response is missing 'filings'")

        return filings

    def fetch_mergers_and_acqusitions(self, name: str) -> list[SecFiling]:
        """
        Fetch SEC docs for mergers and acquisitions based on company name
        """
        return self.fetch_docs(
            [
                f"companyName:{name}",
                'formType:"S-4"',
                'NOT formType:("4/A" OR "S-4 POS")',
            ]
        )
