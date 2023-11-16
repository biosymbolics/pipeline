"""
Handler for patent graph reports
"""
import json
from typing import TypedDict
import logging

from clients import patents as patent_client
from clients.patents.reports.graph import graph_patent_relationships
from handlers.patents.utils import parse_params

from ..types import PatentSearchParams

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ReportEvent(TypedDict):
    queryStringParameters: PatentSearchParams


def graph_patent_characteristics(event: ReportEvent, context):
    """
    Return a graph of patent characteristics

    Invocation:
    - Local: `serverless invoke local --function patents-graph --param='ENV=local' --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - Remote: `serverless invoke --function patents-graph --data='{"queryStringParameters": { "terms":"gpr84 antagonist" }}'`
    - API: `curl https://api.biosymbolics.ai/patents/reports/graph?terms=asthma`
    """
    params = parse_params(event.get("queryStringParameters", {}), default_limit=10000)

    if len(params["terms"]) < 1 or not all([len(t) > 1 for t in params["terms"]]):
        logger.error("Missing or malformed params: %s", params)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching reports for params: %s", params)

    try:
        patents = patent_client.search(**params)
        if len(patents) == 0:
            logging.info("No patents found for terms: %s", params["terms"])
            return {"statusCode": 200, "body": json.dumps({})}

        graph = graph_patent_relationships(patents)
    except Exception as e:
        message = f"Error generating patent reports: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {"statusCode": 200, "body": graph.to_json()}
