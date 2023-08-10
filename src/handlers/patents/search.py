"""
Handler for patents search
"""
import json
from typing_extensions import NotRequired
from typing import TypedDict
import logging

from clients import patents as patent_client
from clients.patents import RelevancyThreshold

logger = logging.getLogger(__name__)


class SearchParams(TypedDict):
    terms: str
    fetch_approval: NotRequired[bool]
    min_patent_years: NotRequired[int]
    relevancy_threshold: NotRequired[RelevancyThreshold]
    max_results: NotRequired[int]


class SearchEvent(TypedDict):
    queryStringParameters: SearchParams


def search(event: SearchEvent, context):
    """
    Search patents by terms

    Invocation:
    - Local: `serverless invoke local --function search-patents --data='{"queryStringParameters": { "terms":["asthma"] }}'`
    - Remote: `serverless invoke --function search-patents --data='{"queryStringParameters": { "terms":["asthma"] }}'`
    - API: `curl https://v8v4ij0xs4.execute-api.us-east-1.amazonaws.com/dev/patents/search?terms=asthma`
    """
    params = event.get("queryStringParameters", {})
    terms = params.get("terms")
    terms_list = terms.split(",") if terms else []

    if not params or not terms or not all([len(t) > 1 for t in terms_list]):
        logging.error(
            "Missing or malformed query params: %s",
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

    logging.info(
        "Fetching patents for terms: %s (%s, %s, %s, %s)",
        terms_list,
        fetch_approval,
        min_patent_years,
        relevancy_threshold,
        max_results,
    )

    patents = patent_client.search(
        terms, fetch_approval, min_patent_years, relevancy_threshold, max_results
    )

    return {"statusCode": 200, "body": json.dumps(patents, default=str)}
