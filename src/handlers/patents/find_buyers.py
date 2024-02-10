"""
Handler for find buyers
"""

import json
import logging
import traceback

from clients.documents import find_buyers as find_patent_buyers
from handlers.utils import handle_async
from typings.client import BuyerFinderParams
from utils.encoding.json_encoder import DataclassJSONEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _find_buyers(raw_event: dict, context):
    """
    Find buyers API

    Invocation:
    - Local: `serverless invoke local --function find-buyers --param='ENV=local' --data='{"queryStringParameters": { "description":"a lipid-based drug delivery platform" }}'`
    - Local: `curl http://localhost:3001/dev/patents/buyers?description=a%20lipid-based%20drug%20delivery%20platform`
    - Local: curl 'http://localhost:3001/dev/patents/buyers?description=a%20lipid-based%20drug%20delivery%20platform&use_gpt_expansion=true'
    """

    p = BuyerFinderParams(**raw_event["queryStringParameters"])

    if p.description is None:
        raise ValueError("Description is required")

    logger.info("Fetching buyers for parameters: %s", p)

    try:
        results = await find_patent_buyers(
            description=p.description, k=p.k, use_gpt_expansion=p.use_gpt_expansion
        )
    except Exception as e:
        message = f"Error finding buyers: {e}"
        logger.error(message)
        traceback.print_exc()
        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(results, cls=DataclassJSONEncoder),
    }


find_buyers = handle_async(_find_buyers)
