"""
Handler for approvals search
"""

import json
import logging

from clients.documents import approval_search
from handlers.utils import handle_async
from typings.client import RegulatoryApprovalSearchParams
from utils.encoding.json_encoder import DataclassJSONEncoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _search(raw_event: dict, context):
    """
    Search trials by terms

    Invocation:
    - Local: `serverless invoke local --function search-approvals --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma", "query_type": "OR" }}'`
    - Remote: `serverless invoke --function search-approvals --data='{"queryStringParameters": { "terms":"pulmonary arterial hypertension" }}'`
    - API: `curl https://api.biosymbolics.ai/approvals/search?terms=asthma`
    """

    p = RegulatoryApprovalSearchParams(**raw_event["queryStringParameters"])

    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching regulatory approvals for params: %s", p)

    try:
        results = await approval_search(p)
    except Exception as e:
        message = f"Error searching regulatory approvals: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(results, cls=DataclassJSONEncoder),
    }


search = handle_async(_search)
