"""
Handler for patents search
"""
import json
from typing import TypedDict
import logging

from clients import patents as patent_client

from .types import PatentSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SearchEvent(TypedDict):
    queryStringParameters: PatentSearchParams


def search(event: SearchEvent, context):
    """
    Search patents by terms

    Invocation:
    - Local: `serverless invoke local --function search-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma" }}'`
    - Local: `serverless invoke local --function search-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"WO-2022076289-A1" }}'`
    - Remote: `serverless invoke --function search-patents --data='{"queryStringParameters": { "terms":"asthma;melanoma" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/search?terms=asthma`
    """

    params = event.get("queryStringParameters", {})
    terms = params.get("terms")
    terms_list = terms.split(";") if terms else []

    if not params or not terms or not all([len(t) > 1 for t in terms_list]):
        logger.error("Missing or malformed params: %s", params)
        return {"statusCode": 400, "message": "Missing params(s)"}

    fetch_approval = params.get("fetch_approval") or True
    min_patent_years = params.get("min_patent_years") or 10
    relevancy_threshold = params.get("relevancy_threshold") or "high"
    max_results = params.get("max_results") or 1000

    logger.info("Fetching patents for params: %s", params)

    try:
        results = patent_client.search(
            terms_list,
            fetch_approval,
            min_patent_years,
            relevancy_threshold,
            max_results,
        )

    except Exception as e:
        logger.error("Error searching patents: %s (%s)", e, str(type(e)))
        return {"statusCode": 500, "message": str(e)}

    return {"statusCode": 200, "body": json.dumps(results, default=str)}
