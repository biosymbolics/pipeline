"""
Handler for patents search
"""
import json
from typing_extensions import NotRequired
from typing import TypedDict
import logging

from clients import patents as patent_client
from clients.patents import RelevancyThreshold


class SearchParams(TypedDict):
    terms: list[str]
    fetch_approval: NotRequired[bool]
    min_patent_years: NotRequired[int]
    relevancy_threshold: NotRequired[RelevancyThreshold]
    max_results: NotRequired[int]


class SearchEvent(TypedDict):
    query: SearchParams


def search(event: SearchEvent, context):
    """
    Search patents by terms

    Invocation:
    - Local: `serverless invoke local --function search-patents --data='{"query": { "terms":["asthma"] }}'`
    - Remote: `serverless invoke --function search-patents --data='{"query": { "terms":["asthma"] }}'`
    - API: `curl https://v8v4ij0xs4.execute-api.us-east-1.amazonaws.com/dev/patents/search?terms=asthma`
    """
    params = event.get("query", {})
    terms = params.get("terms")

    if not params or not terms:
        logging.error(
            "Missing query or param `terms`, params: %s",
            params,
        )
        return {
            "statusCode": 400,
            "message": "Missing parameter(s)",
        }

    fetch_approval = params.get("fetch_approval") or False
    min_patent_years = params.get("min_patent_years") or 10
    relevancy_threshold = params.get("relevancy_threshold") or "high"
    max_results = params.get("max_results") or 100

    patents = patent_client.search(
        terms, fetch_approval, min_patent_years, relevancy_threshold, max_results
    )

    return {"statusCode": 200, "body": json.dumps(patents, default=str)}
