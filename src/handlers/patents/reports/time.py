"""
Handler for patent timewise reports
"""
import json
from typing import TypedDict
import logging

from clients import patents as patent_client
from clients.patents.constants import DOMAINS_OF_INTEREST
from clients.patents.reports import aggregate

from ..types import PatentSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportEvent(TypedDict):
    queryStringParameters: PatentSearchParams


def aggregate_over_time(event: ReportEvent, context):
    """
    Aggregate patents trends over time

    Invocation:
    - Local: `serverless invoke local --function patents-over-time --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma" }}'`
    - Remote: `serverless invoke --function patents-over-time --data='{"queryStringParameters": { "terms":"asthma;melanoma" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/time?terms=asthma`
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
    max_results = params.get("max_results") or 10000  # higher limit for reports

    try:
        patents = patent_client.search(
            terms_list,
            fetch_approval,
            min_patent_years,
            relevancy_threshold,
            max_results,
        )
        if len(patents) == 0:
            logging.info("No patents found for terms: %s", terms)
            return {"statusCode": 200, "body": json.dumps([])}

        summaries = aggregate(
            patents,
            x_dimensions=[*DOMAINS_OF_INTEREST, "ipc_codes", "similar"],
            y_dimensions=["priority_date"],
            y_transform=lambda y: y.year,
        )
    except Exception as e:
        logger.error("Error generating reports for patents: %s (%s)", e, str(type(e)))
        return {"statusCode": 500, "message": str(e)}

    return {"statusCode": 200, "body": json.dumps(summaries, default=str)}
