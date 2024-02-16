"""
Handler for entities search
"""

import json
import logging
import traceback

from clients.documents import asset_search
from handlers.utils import handle_async
from typings.client import AssetSearchParams
from utils.encoding.json_encoder import DataclassJSONEncoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _search(raw_event: dict, context):
    """
    Search assets by terms

    Invocation:
    - Local: `serverless invoke local --function search-assets --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma", "query_type": "OR", "limit": 50 }}'`
    - Remote: `serverless invoke --function search-assets --data='{"queryStringParameters": { "terms":"pulmonary arterial hypertension", "limit": 50 }}'`
    - API: `curl https://api.biosymbolics.ai/assets/search?terms=asthma`
    """
    p = AssetSearchParams(
        **(raw_event.get("queryStringParameters") or {}),
    )

    if (
        len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms])
    ) and p.description is None:
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching assets for params: %s", p)

    try:
        assets = await asset_search(p)
    except Exception as e:
        message = f"Error searching entities: {e}"
        logger.error(message)
        traceback.print_exc()

        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(assets, cls=DataclassJSONEncoder),
    }


search = handle_async(_search)
