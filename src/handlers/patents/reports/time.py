"""
Handler for patent timewise reports
"""
import json
import logging
from pydantic import BaseModel

from clients import patents as patent_client
from clients.patents.constants import DOMAINS_OF_INTEREST
from clients.patents.reports import group_by_xy
from handlers.patents.utils import parse_params
from typings.client import (
    RawPatentSearchParams,
    OptionalRawPatentSearchParams as OptionalParams,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportEvent(BaseModel):
    queryStringParameters: RawPatentSearchParams


def aggregate_over_time(raw_event: dict, context):
    """
    Aggregate patents trends over time

    Invocation:
    - Local: `serverless invoke local --function patents-over-time --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma" }}'`
    - Remote: `serverless invoke --function patents-over-time --data='{"queryStringParameters": { "terms":"asthma" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/time?terms=asthma`
    """
    event = ReportEvent(**raw_event)
    p = parse_params(
        event.queryStringParameters,
        OptionalParams(term_field="category_rollup"),
        10000,
    )

    if not p or len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports forparams: %s", p)

    try:
        patents = patent_client.search(p)
        if len(patents) == 0:
            logging.info("No patents found for terms: %s", p.terms)
            return {"statusCode": 200, "body": json.dumps([])}

        summaries = group_by_xy(
            patents,
            x_dimensions=[*DOMAINS_OF_INTEREST, "similar_patents"],
            y_dimensions=["priority_date"],
            y_transform=lambda y: y.year,
        )
    except Exception as e:
        message = f"Error generating patent reports: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(summaries, default=str)}
