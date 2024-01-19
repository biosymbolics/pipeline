"""
Handler for patent timewise reports
"""
import json
import logging

from clients.documents.reports import XYReport
from handlers.utils import handle_async
from typings.client import DocumentSearchParams, PatentSearchParams

from .constants import DEFAULT_REPORT_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class XByYReportParams(PatentSearchParams):
    x: str
    y: str


async def _x_by_y(raw_event: dict, context):
    """
    Get an x by y report

    Invocation:
    - Local: `serverless invoke local --function documents-x-by-y --param='ENV=local' --data='{"queryStringParameters": { "x": "assignees", "y": "diseases", "terms":"asthma" }}'`
    - Remote: `serverless invoke --function documents-x-by-y --data='{"queryStringParameters": { "x": "assignees", "y": "diseases", "terms":"asthma" }}'`
    - API: `curl 'https://api.biosymbolics.ai/reports/x_by_y?terms=asthma&x=assignees&y=diseases'`
    """
    p = XByYReportParams(
        **{**raw_event["queryStringParameters"], **DEFAULT_REPORT_PARAMS}
    )

    x_dimension = p.x
    y_dimension = p.y

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
        report = await XYReport.group_by_xy(
            search_params=DocumentSearchParams(terms=p.terms, query_type=p.query_type),
            x_dimension=x_dimension,
            y_dimension=y_dimension,
        )

    except Exception as e:
        message = f"Error generating patent reports: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(report, default=str)}


x_by_y = handle_async(_x_by_y)
