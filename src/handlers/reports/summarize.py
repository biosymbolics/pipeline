"""
Handler for patent summarization reports
"""
import json
import logging

from clients.documents.patents.constants import DOMAINS_OF_INTEREST
from clients.documents.reports import group_by_xy_for_filters
from handlers.utils import handle_async
from utils.encoding.json_encoder import DataclassJSONEncoder
from typings.client import CommonSearchParams

from .constants import DEFAULT_REPORT_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _summarize(raw_event: dict, context):
    """
    Summarize patents by terms (diseases, compounds, etc)

    Invocation:
    - Local: `serverless invoke local --function summarize-documents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma",  "skip_cache": true }}'`
    - Local: `serverless invoke local --function summarize-documents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma",  "term_field": "canonical_name" }}'`
    - Remote: `serverless invoke --function summarize-documents --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/summarize?terms=asthma`
    """
    p = CommonSearchParams(
        **{**raw_event["queryStringParameters"], **DEFAULT_REPORT_PARAMS}
    )

    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", p)

    try:
        summaries = await group_by_xy_for_filters(
            search_params=p,
            x_dimension="canonical_name",
            filters=[f"canonical_type in ('{d}')" for d in DOMAINS_OF_INTEREST],
        )
    except Exception as e:
        message = f"Error reporting on patents: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(summaries, cls=DataclassJSONEncoder)}


summarize = handle_async(_summarize)
