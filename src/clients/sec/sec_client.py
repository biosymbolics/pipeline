"""
Client for SEC API
"""

import logging
import os
from typing import Sequence
from sec_api import QueryApi
from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder

from utils.string import create_hash_key, get_id

from .types import SecFiling


API_KEY = os.environ.get("SEC_API_KEY") or ""


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

    async def fetch_docs(
        self, criteria: list[str], take: int = 100, skip: int = 0
    ) -> list[SecFiling]:
        """
        Fetch SEC docs based on specified criteria
        e.g. ["ticker:PFE", "filedAt:{2020-01-01 TO 2020-12-31}", "formType:10-K"] -> docs

        Uses s3 cache.
        """
        query = self._get_query(criteria, take, skip)
        logging.info("Getting SEC docs with query %s", query)

        async def fetch():
            return self.client.get_filings(query)

        key = get_id({"criteria": create_hash_key(criteria), "api": "sec"})
        response = await retrieve_with_cache_check(
            fetch,
            key=key,
            decode=lambda str_data: storage_decoder(str_data),
            cache_name="biosym-etl-cache",
            use_filesystem=True,
        )

        if not response:
            raise Exception("No response from SEC API")

        filings = [SecFiling(**f) for f in response.get("filings")]

        if not filings:
            logging.error("Response is missing 'filings': %s", response)

        return filings or []

    async def fetch_mergers_and_acquisitions(
        self, symbols: Sequence[str]
    ) -> dict[str, list[SecFiling]]:
        """
        Fetch SEC docs for mergers and acquisitions based on ticker symbol
        """

        async def fetch(symbol: str):
            return await self.fetch_docs(
                [
                    f"ticker:{symbol}",
                    'formType:"S-4"',
                    'NOT formType:("4/A" OR "S-4 POS")',
                ],
                take=100,
            )

        return {symbol: await fetch(symbol) for symbol in symbols}
