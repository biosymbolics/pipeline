"""
Handler for patent timewise reports
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


def aggregate_over_time(event: ReportEvent, context):
    """
    Aggregate patents trends over time

    Invocation:
    - Local: `serverless invoke local --function patents-over-time --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma" }}'`
    - Remote: `serverless invoke --function patents-over-time --data='{"queryStringParameters": { "terms":"asthma;melanoma" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/time?terms=asthma`
    """
    params = parse_params(event.get("queryStringParameters", {}), 10000)

    if (
        not params
        or len(params["terms"]) < 1
        or not all([len(t) > 1 for t in params["terms"]])
    ):
        logger.error("Missing or malformed params: %s", params)
        return {"statusCode": 400, "message": "Missing params(s)"}

    logger.info("Fetching reports forparams: %s", params)

    try:
        patents = patent_client.search(**params)
        if len(patents) == 0:
            logging.info("No patents found for terms: %s", params["terms"])
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
