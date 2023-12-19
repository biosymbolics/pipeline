"""
Handler for trials search
"""
import json
import logging

from clients import trials as trial_client
from typings.client import TrialSearchParams
from utils.encoding.json_encoder import DataclassJSONEncoder


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def search(raw_event: dict, context):
    """
    Search trials by terms

    Invocation:
    - Local: `serverless invoke local --function search-trials --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma", "query_type": "OR" }}'`
    - Remote: `serverless invoke --function search-trials --data='{"queryStringParameters": { "terms":"pulmonary arterial hypertension" }}'`
    - API: `curl https://api.biosymbolics.ai/trials/search?terms=asthma`
    """

    p = TrialSearchParams(*raw_event["queryStringParameters"])

    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching trials for params: %s", p)

    try:
        results = trial_client.search(p)
    except Exception as e:
        message = f"Error searching trials: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(results, cls=DataclassJSONEncoder),
    }
