"""
Handler for patent timewise reports
"""
import json
import logging

from clients.documents import patents as patent_client
from clients.documents.patents.constants import DOMAINS_OF_INTEREST
from clients.documents.patents.reports import group_by_xy
from handlers.patents.reports.constants import DEFAULT_REPORT_PARAMS
from handlers.utils import handle_async
from typings.client import PatentSearchParams
from utils.encoding.json_encoder import DataclassJSONEncoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _aggregate_over_time(raw_event: dict, context):
    """
    Aggregate patents trends over time

    Invocation:
    - Local: `serverless invoke local --function patents-over-time --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma" }}'`
    - Remote: `serverless invoke --function patents-over-time --data='{"queryStringParameters": { "terms":"asthma" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/time?terms=asthma`
    """
    p = PatentSearchParams(
        **{**raw_event["queryStringParameters"], **DEFAULT_REPORT_PARAMS}
    )

    if not p or len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports forparams: %s", p)

    try:
        patents = await patent_client.search(p)
        if len(patents) == 0:
            logging.info("No patents found for terms: %s", p.terms)
            return {"statusCode": 200, "body": json.dumps([])}

        summaries = group_by_xy(
            patents,
            x_dimensions=DOMAINS_OF_INTEREST,
            y_dimensions=["priority_date"],
            y_transform=lambda y: y.year,
        )
    except Exception as e:
        message = f"Error generating patent reports: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(summaries, cls=DataclassJSONEncoder)}


aggregate_over_time = handle_async(_aggregate_over_time)
