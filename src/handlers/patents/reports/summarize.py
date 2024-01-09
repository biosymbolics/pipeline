"""
Handler for patent summarization reports
"""
import json
import logging

from clients.documents import patents as patent_client
from clients.documents.patents.constants import DOMAINS_OF_INTEREST
from clients.documents.patents.reports.reports import group_by_xy
from handlers.utils import handle_async
from utils.encoding.json_encoder import DataclassJSONEncoder
from typings.client import PatentSearchParams

from .constants import DEFAULT_REPORT_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _summarize(raw_event: dict, context):
    """
    Summarize patents by terms (diseases, compounds, etc)

    Invocation:
    - Local: `serverless invoke local --function summarize-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma",  "skip_cache": true }}'`
    - Local: `serverless invoke local --function summarize-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma",  "term_field": "instance_rollup" }}'`
    - Remote: `serverless invoke --function summarize-patents --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/summarize?terms=asthma`
    """
    p = PatentSearchParams(
        **{**raw_event["queryStringParameters"], **DEFAULT_REPORT_PARAMS}
    )

    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", p)

    try:
        results = await patent_client.search(p)
        summaries = group_by_xy(results, [*DOMAINS_OF_INTEREST, "similar_patents"])
    except Exception as e:
        message = f"Error reporting on patents: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(summaries, cls=DataclassJSONEncoder)}


summarize = handle_async(_summarize)
