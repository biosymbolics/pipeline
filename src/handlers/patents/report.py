"""
Handler for patents search
"""
import json
from typing import TypedDict
import logging

from clients import patents as patent_client
from clients.patents.constants import DOMAINS_OF_INTEREST
from clients.patents.reports import generate_summaries

from .types import PatentSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportEvent(TypedDict):
    queryStringParameters: PatentSearchParams


def summarize(event: ReportEvent, context):
    """
    Summarize patents by terms (diseases, compounds, etc)

    Invocation:
    - Local: `serverless invoke local --function summarize-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma" }}'`
    - Remote: `serverless invoke --function summarize-patents --data='{"queryStringParameters": { "terms":"asthma;melanoma" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/search?terms=asthma`
    """
    params = event.get("queryStringParameters", {})
    terms = params.get("terms")
    terms_list = terms.split(";") if terms else []

    if not params or not terms or not all([len(t) > 1 for t in terms_list]):
        logger.error("Missing or malformed params: %s", params)
        return {"statusCode": 400, "message": "Missing params(s)"}

    logger.info("Fetching reports forparams: %s", params)

    fetch_approval = params.get("fetch_approval") or True
    min_patent_years = params.get("min_patent_years") or 10
    relevancy_threshold = params.get("relevancy_threshold") or "high"
    max_results = params.get("max_results") or 1000

    try:
        results = patent_client.search(
            terms_list,
            fetch_approval,
            min_patent_years,
            relevancy_threshold,
            max_results,
        )
        summaries = generate_summaries(
            results, [*DOMAINS_OF_INTEREST, "ipc_codes", "similar"]
        )
    except Exception as e:
        logger.error("Error generating reports for patents: %s (%s)", e, str(type(e)))
        return {"statusCode": 500, "message": str(e)}

    return {"statusCode": 200, "body": json.dumps(summaries, default=str)}
