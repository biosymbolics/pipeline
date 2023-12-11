"""
Handler for patent summarization reports
"""
import json
import logging
from pydantic import BaseModel

from clients import patents as patent_client
from clients.patents.constants import DOMAINS_OF_INTEREST
from clients.patents.reports.reports import group_by_xy
from handlers.patents.utils import parse_params
from typings.client import RawPatentSearchParams

from .constants import DEFAULT_REPORT_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportEvent(BaseModel):
    queryStringParameters: RawPatentSearchParams


def summarize(raw_event: dict, context):
    """
    Summarize patents by terms (diseases, compounds, etc)

    Invocation:
    - Local: `serverless invoke local --function summarize-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma",  "skip_cache": true }}'`
    - Local: `serverless invoke local --function summarize-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma",  "term_field": "instance_rollup" }}'`
    - Remote: `serverless invoke --function summarize-patents --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/summarize?terms=asthma`
    """
    event = ReportEvent(**raw_event)
    p = parse_params(event.queryStringParameters, DEFAULT_REPORT_PARAMS, 10000)

    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", p)

    try:
        results = patent_client.search(p)
        summaries = group_by_xy(results, [*DOMAINS_OF_INTEREST, "similar_patents"])
    except Exception as e:
        message = f"Error reporting on patents: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(summaries, default=str)}
