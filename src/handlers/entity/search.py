"""
Handler for entities search
"""

import json
import logging
import traceback

from clients.documents import entity_search
from handlers.utils import handle_async
from typings.client import EntitySearchParams
from utils.encoding.json_encoder import DataclassJSONEncoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _search(raw_event: dict, context):
    """
    Search entities by terms

    Invocation:
    - Local: `serverless invoke local --function search-entities --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma", "query_type": "OR", "limit": 50 }}'`
    - Remote: `serverless invoke --function search-entities --data='{"queryStringParameters": { "terms":"pulmonary arterial hypertension", "limit": 50 }}'`
    - API: `curl https://api.biosymbolics.ai/entities/search?terms=asthma`
    """
    p = EntitySearchParams(
        **(raw_event.get("queryStringParameters") or {}),
    )

    if (
        len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms])
    ) and p.description is None:
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching entities for params: %s", p)

    try:
        entities = await entity_search(p)
    except Exception as e:
        message = f"Error searching entities: {e}"
        logger.error(message)
        traceback.print_exc()
        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(entities, cls=DataclassJSONEncoder),
    }


search = handle_async(_search)
