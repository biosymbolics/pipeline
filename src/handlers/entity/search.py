"""
Handler for entities search
"""
import json
import logging
from pydantic import BaseModel

from clients import entity as entity_client
from typings.client import RawEntitySearchParams
from utils.encoding.json_encoder import DataclassJSONEncoder

from .utils import parse_params

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class EntitySearchEvent(BaseModel):
    queryStringParameters: RawEntitySearchParams


def search(raw_event: dict, context):
    """
    Search entities by terms

    Invocation:
    - Local: `serverless invoke local --function search-entities --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma", "query_type": "OR" }}'`
    - Remote: `serverless invoke --function search-entities --data='{"queryStringParameters": { "terms":"pulmonary arterial hypertension" }}'`
    - API: `curl https://api.biosymbolics.ai/entities/search?terms=asthma`
    """

    event = EntitySearchEvent(**raw_event)
    p = parse_params(event.queryStringParameters)

    if len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms]):
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching entities for params: %s", p)

    try:
        results = entity_client.search(p)
    except Exception as e:
        message = f"Error searching entities: {e}"
        logger.error(message)
        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(results, cls=DataclassJSONEncoder),
    }
