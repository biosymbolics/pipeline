"""
Handler for patent timewise reports
"""
import json
import logging

from clients.documents import patents as patent_client
from clients.documents.patents.reports.reports import group_by_xy
from handlers.utils import handle_async
from typings.client import PatentSearchParams

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
    - Local: `serverless invoke local --function patents-x-by-y --param='ENV=local' --data='{"queryStringParameters": { "x": "assignees", "y": "diseases", "terms":"asthma" }}'`
    - Remote: `serverless invoke --function patents-x-by-y --data='{"queryStringParameters": { "x": "assignees", "y": "diseases", "terms":"asthma" }}'`
    - API: `curl 'https://api.biosymbolics.ai/patents/reports/x_by_y?terms=asthma&x=assignees&y=diseases'`
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
        patents = await patent_client.search(p)
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


x_by_y = handle_async(_x_by_y)
