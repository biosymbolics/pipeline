"""
Handler for patents search
"""
import json
from typing import TypedDict
import logging

from clients import patents as patent_client
from handlers.patents.utils import parse_params

from .types import PatentSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class SearchEvent(TypedDict):
    queryStringParameters: PatentSearchParams


def search(event: SearchEvent, context):
    """
    Search patents by terms

    Invocation:
    - Local: `serverless invoke local --function search-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma", "domains": "compounds" }}'`
    - Local: `serverless invoke local --function search-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma", "domains": "diseases" }}'`
    - Local: `serverless invoke local --function search-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"WO-2022076289-A1" }}'`
    - Remote: `serverless invoke --function search-patents --data='{"queryStringParameters": { "terms":"asthma;melanoma" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/search?terms=asthma`
    """

    params = parse_params(event.get("queryStringParameters", {}))

    if (
        not params
        or len(params["terms"]) < 1
        or not all([len(t) > 1 for t in params["terms"]])
    ):
        logger.error("Missing or malformed params: %s", params)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching patents for params: %s", params)

    try:
        results = patent_client.search(**params)

    except Exception as e:
        message = f"Error searching patents: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(results, default=str)}
