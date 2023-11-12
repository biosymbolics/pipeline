"""
Handler for patent summarization reports
"""
import json
from typing import TypedDict
import logging

from clients import patents as patent_client
from clients.patents.constants import DOMAINS_OF_INTEREST
from clients.patents.reports import group_by_xy
from handlers.patents.utils import parse_params

from .constants import DEFAULT_REPORT_PARAMS
from ..types import PatentSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportEvent(TypedDict):
    queryStringParameters: PatentSearchParams


def summarize(event: ReportEvent, context):
    """
    Summarize patents by terms (diseases, compounds, etc)

    Invocation:
    - Local: `serverless invoke local --function summarize-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma",  "skip_cache": true }}'`
    - Local: `serverless invoke local --function summarize-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma",  "term_field": "rollup_term" }}'`
    - Remote: `serverless invoke --function summarize-patents --data='{"queryStringParameters": { "terms":"asthma" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/summarize?terms=asthma`
    """
    params = parse_params(
        event.get("queryStringParameters", {}), DEFAULT_REPORT_PARAMS, 10000
    )

    if (
        not params
        or len(params["terms"]) < 1
        or not all([len(t) > 1 for t in params["terms"]])
    ):
        logger.error("Missing or malformed params: %s", params)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", params)

    try:
        results = patent_client.search(**params)
        summaries = group_by_xy(results, [*DOMAINS_OF_INTEREST, "similar_patents"])
    except Exception as e:
        message = f"Error reporting on patents: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(summaries, default=str)}
