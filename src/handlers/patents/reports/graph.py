"""
Handler for patent graph reports
"""
import json
import logging
from pydantic import BaseModel

from clients import patents as patent_client
from clients.patents.reports.graph import graph_patent_relationships
from handlers.patents.reports.constants import DEFAULT_REPORT_PARAMS
from handlers.patents.utils import parse_params
from typings.client import RawPatentSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportEvent(BaseModel):
    queryStringParameters: RawPatentSearchParams


def graph_patent_characteristics(raw_event: dict, context):
    """
    Return a graph of patent characteristics

    Invocation:
    - Local: `serverless invoke local --function patents-graph --param='ENV=local' --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - Remote: `serverless invoke --function patents-graph --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/graph?terms=asthma`
    """
    p = parse_params(raw_event["queryStringParameters"], DEFAULT_REPORT_PARAMS)

    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", p)

    try:
        patents = patent_client.search(p)
        if len(patents) == 0:
            logging.info("No patents found for terms: %s", p.terms)
            return {"statusCode": 200, "body": json.dumps({})}

        graph = graph_patent_relationships(patents)
    except Exception as e:
        message = f"Error generating patent reports: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": graph.to_json()}
