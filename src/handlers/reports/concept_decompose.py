"""
Handler for concept decomposition reports
"""

import json
import logging
import traceback

from clients.vector import ConceptDecomposer
from handlers.utils import handle_async
from typings.client import ConceptDecomposeParams
from utils.encoding.json_encoder import DataclassJSONEncoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _decompose_concepts(raw_event: dict, context):
    """
    Aggregate patents trends over time

    Invocation:
    - Local: `serverless invoke local --function documents-over-time --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma" }}'`
    - Remote: `serverless invoke --function documents-over-time --data='{"queryStringParameters": { "terms":"asthma" }}'`
    - API: `curl https://api.biosymbolics.ai/reports/time?terms=asthma`
    """
    p = ConceptDecomposeParams(**{**raw_event["queryStringParameters"]})

    logger.info("Fetching reports for params: %s", p)

    try:
        report = await ConceptDecomposer().decompose_concept_with_reports(p.description)
    except Exception as e:
        message = f"Error generating patent reports: {e}"
        logger.error(message)
        traceback.print_exc()
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(report, cls=DataclassJSONEncoder)}


decompose_concepts = handle_async(_decompose_concepts)
