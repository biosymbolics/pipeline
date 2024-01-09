"""
Handler for entities search
"""
import json
import logging

from clients.documents import asset_search
from typings.client import AssetSearchParams
from utils.encoding.json_encoder import DataclassJSONEncoder

from .constants import DEFAULT_SEARCH_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def search(raw_event: dict, context):
    """
    Search assets by terms

    Invocation:
    - Local: `serverless invoke local --function search-assets --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma", "query_type": "OR" }}'`
    - Remote: `serverless invoke --function search-assets --data='{"queryStringParameters": { "terms":"pulmonary arterial hypertension" }}'`
    - API: `curl https://api.biosymbolics.ai/assets/search?terms=asthma`
    """
    p = AssetSearchParams(
        **{
            **(raw_event.get("queryStringParameters") or {}),
            **DEFAULT_SEARCH_PARAMS,
        },
    )

    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching assets for params: %s", p)

    try:
        results = asset_search(p)
    except Exception as e:
        message = f"Error searching entities: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(results, cls=DataclassJSONEncoder),
    }
