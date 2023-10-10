"""
Handler for patent timewise reports
"""
import json
from typing import TypedDict
import logging

from clients import patents as patent_client
from clients.patents.reports import aggregate
from handlers.patents.utils import parse_params

from ..types import PatentSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class XByYReportParams(PatentSearchParams):
    x: str
    y: str


class XByYReportEvent(TypedDict):
    queryStringParameters: XByYReportParams


def x_by_y(event: XByYReportEvent, context):
    """
    Get an x by y report

    Invocation:
    - Local: `serverless invoke local --function patents-x-by-y --param='ENV=local' --data='{"queryStringParameters": { "x": "assignees", "y": "diseases", "terms":"asthma" }}'`
    - Remote: `serverless invoke --function patents-x-by-y --data='{"queryStringParameters": { "x": "assignees", "y": "diseases", "terms":"asthma" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/x_by_y?terms=asthma&x=assignees&y=diseases`
    """
    params = parse_params(event.get("queryStringParameters", {}), 10000)
    x_dimension = event.get("queryStringParameters", {}).get("x")
    y_dimension = event.get("queryStringParameters", {}).get("y")

    if (
        not params
        or len(params["terms"]) < 1
        or not all([len(t) > 1 for t in params["terms"]])
        or x_dimension is None
        or y_dimension is None
    ):
        logger.error("Missing or malformed params: %s", params)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", params)

    try:
        patents = patent_client.search(**params)
        if len(patents) == 0:
            logging.info("No patents found for terms: %s", params["terms"])
            return {"statusCode": 200, "body": json.dumps([])}

        reports = aggregate(
            patents,
            x_dimensions=[x_dimension],
            y_dimensions=[y_dimension],
        )

        report = reports[0]
    except Exception as e:
        message = f"Error generating patent reports: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(report, default=str)}
