"""
Handler for patent topic reports
"""
import json
import logging
from pydantic import BaseModel

from clients import patents as patent_client
from clients.patents.reports.topic import model_patent_topics
from handlers.patents.utils import parse_params
from typings.client import RawPatentSearchParams

from .constants import DEFAULT_REPORT_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportEvent(BaseModel):
    queryStringParameters: RawPatentSearchParams


def analyze_topics(raw_event: dict, context):
    """
    Analyize patent topics

    Invocation:
    - Local: `serverless invoke local --function patent-topics --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma",  "skip_cache": true }}'`
    - Remote: `serverless invoke --function patent-topics --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/topics?terms=asthma`
    """
    p = parse_params(raw_event["queryStringParameters"], DEFAULT_REPORT_PARAMS)

    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", p)

    try:
        results = patent_client.search(p)
        topics = model_patent_topics(results)
    except Exception as e:
        message = f"Error reporting on patents: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": json.dumps(topics, default=str)}
