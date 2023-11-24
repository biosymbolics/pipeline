"""
Handler for patent timewise reports
"""
import json
import logging
from pydantic import BaseModel

from clients import patents as patent_client
from clients.patents.reports.reports import group_by_xy
from handlers.patents.utils import parse_params

from .constants import DEFAULT_REPORT_PARAMS
from ..types import RawPatentSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class XByYReportParams(RawPatentSearchParams):
    x: str
    y: str


class XByYReportEvent(BaseModel):
    queryStringParameters: XByYReportParams


def x_by_y(raw_event: dict, context):
    """
    Get an x by y report

    Invocation:
    - Local: `serverless invoke local --function patents-x-by-y --param='ENV=local' --data='{"queryStringParameters": { "x": "assignees", "y": "diseases", "terms":"asthma" }}'`
    - Remote: `serverless invoke --function patents-x-by-y --data='{"queryStringParameters": { "x": "assignees", "y": "diseases", "terms":"asthma" }}'`
    - API: `curl 'https://api.biosymbolics.ai/patents/reports/x_by_y?terms=asthma&x=assignees&y=diseases'`
    """
    event = XByYReportEvent(**raw_event)
    p = parse_params(event.queryStringParameters, DEFAULT_REPORT_PARAMS, 10000)
    x_dimension = event.queryStringParameters.x
    y_dimension = event.queryStringParameters.y

    if (
        not p
        or len(p.terms) < 1
        or not all([len(t) > 1 for t in p.terms])
        or x_dimension is None
        or y_dimension is None
    ):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", p)

    try:
        patents = patent_client.search(p)
        if len(patents) == 0:
            logging.info("No patents found for terms: %s", p.terms)
            return {"statusCode": 200, "body": json.dumps([])}

        reports = group_by_xy(
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
