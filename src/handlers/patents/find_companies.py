"""
Handler for find companies
"""

import json
import logging
import traceback

from clients.documents import find_companies_semantically
from handlers.utils import handle_async
from typings.client import CompanyFinderParams
from utils.encoding.json_encoder import DataclassJSONEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _find_companies(raw_event: dict, context):
    """
    Find companies (semantically) API
    TODO: improve clarity about what this does vs owner-rolled-up results

    Invocation:
    - Local: `serverless invoke local --function find-companies --param='ENV=local' --data='{"queryStringParameters": { "description":"a lipid-based drug delivery platform" }}'`
    - Local: `curl http://localhost:3001/dev/patents/companies?description=a%20lipid-based%20drug%20delivery%20platform`
    - Local: curl 'http://localhost:3001/dev/patents/companies?description=a%20lipid-based%20drug%20delivery%20platform&use_gpt_expansion=true'
    """

    p = CompanyFinderParams(**raw_event["queryStringParameters"])

    if p.description is None and len(p.similar_companies) == 0:
        raise ValueError("Description and/or similar_companies are required")

    logger.info("Fetching companies for parameters: %s", p)

    try:
        results = await find_companies_semantically(p)
    except Exception as e:
        message = f"Error finding companies: {e}"
        logger.error(message)
        traceback.print_exc()
        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(results, cls=DataclassJSONEncoder),
    }


find_companies = handle_async(_find_companies)
