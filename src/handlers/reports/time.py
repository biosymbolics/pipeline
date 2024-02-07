"""
Handler for patent timewise reports
"""

import json
import logging
import traceback

from clients.documents.reports import XYReport
from handlers.utils import handle_async
from typings.client import DocumentSearchParams
from typings import DOMAINS_OF_INTEREST
from utils.encoding.json_encoder import DataclassJSONEncoder


from .constants import DEFAULT_REPORT_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _aggregate_over_time(raw_event: dict, context):
    """
    Aggregate patents trends over time

    Invocation:
    - Local: `serverless invoke local --function documents-over-time --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma" }}'`
    - Remote: `serverless invoke --function documents-over-time --data='{"queryStringParameters": { "terms":"asthma" }}'`
    - API: `curl https://api.biosymbolics.ai/reports/time?terms=asthma`
    """
    p = DocumentSearchParams(
        **{**raw_event["queryStringParameters"], **DEFAULT_REPORT_PARAMS}
    )

    if not p or len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", p)

    try:
        summaries = await XYReport.group_by_xy_for_filters(
            filters={d: f"type in ('{d}')" for d in DOMAINS_OF_INTEREST},
            search_params=p,
            x_dimension="canonical_name",  # keyof typeof X_DIMENSIONS
            y_dimension="priority_date",
            limit=10,
            limit_dimension="x",
        )
    except Exception as e:
        message = f"Error generating patent reports: {e}"
        logger.error(message)
        traceback.print_exc()
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(summaries, cls=DataclassJSONEncoder)}


aggregate_over_time = handle_async(_aggregate_over_time)
