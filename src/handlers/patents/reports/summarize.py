"""
Handler for patent summarization reports
"""
import json
from typing import TypedDict
import logging

from clients import patents as patent_client
from clients.patents.constants import DOMAINS_OF_INTEREST
from clients.patents.reports import aggregate
from handlers.patents.utils import parse_params

from ..types import PatentSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportEvent(TypedDict):
    queryStringParameters: PatentSearchParams


def summarize(event: ReportEvent, context):
    """
    Summarize patents by terms (diseases, compounds, etc)

    Invocation:
    - Local: `serverless invoke local --function summarize-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma",  "skip_cache": "True" }}'`
    - Remote: `serverless invoke --function summarize-patents --data='{"queryStringParameters": { "terms":"asthma" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/summarize?terms=asthma`
    """
    params = parse_params(event.get("queryStringParameters", {}), 10000)

    if (
        not params
        or len(params["terms"]) < 1
        or not all([len(t) > 1 for t in params["terms"]])
    ):
        logger.error("Missing or malformed params: %s", params)
        return {"statusCode": 400, "message": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", params)

    try:
        results = patent_client.search(**params)
        summaries = aggregate(results, [*DOMAINS_OF_INTEREST, "ipc_codes", "similar"])
    except Exception as e:
        logger.error("Error generating reports for patents: %s (%s)", e, str(type(e)))
        return {"statusCode": 500, "message": str(e)}

    return {"statusCode": 200, "body": json.dumps(summaries, default=str)}
