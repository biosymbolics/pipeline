"""
Handler for patents search
"""

import json
import logging
import traceback

from clients.documents import patent_search
from handlers.utils import handle_async
from typings.client import PatentSearchParams
from utils.encoding.json_encoder import DataclassJSONEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _search(raw_event: dict, context):
    """
    Search patents by terms

    Invocation:
    - Local: `serverless invoke local --function search-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"asthma;melanoma", "query_type": "OR" }}'`
    - Local: `serverless invoke local --function search-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"presbyopia" }}'`
    - Local: `serverless invoke local --function search-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"melanoma", "term_field": "instance_rollup" }}'`
    - Local: `serverless invoke local --function search-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"pulmonary arterial hypertension", "skip_cache": true }}'`
    - Local: `serverless invoke local --function search-patents --param='ENV=local' --data='{"queryStringParameters": { "terms":"WO-2022076289-A1", "skip_cache": true }}'`
    - Remote: `serverless invoke --function search-patents --data='{"queryStringParameters": { "terms":"pulmonary arterial hypertension" }}'`
    - Local: `curl 'http://localhost:3001/dev/patents/search?description=a%20lipid-based%20nanoparticle%20drug%20delivery%20system'`
    - API: `curl https://api.biosymbolics.ai/patents/search?terms=asthma`
    - API: `curl https://api.biosymbolics.ai/patents/search?terms=WO-2022076289-A1`
    """

    p = PatentSearchParams(**raw_event["queryStringParameters"])

    if p.include is None:
        raise ValueError("Include is required")

    if (
        len(p.terms) < 1 or not all([len(t) > 1 for t in p.terms])
    ) and p.description is None:
        logger.error("Missing or malformed params: %s", p)
        return {"statusCode": 400, "body": "Missing params(s)"}

    logger.info("Fetching patents for params: %s", p)

    try:
        results = await patent_search(p)
    except Exception as e:
        message = f"Error searching patents: {e}"
        logger.error(message)
        traceback.print_exc()
        return {"statusCode": 500, "body": message}

    return {
        "statusCode": 200,
        "body": json.dumps(results, cls=DataclassJSONEncoder),
    }


search = handle_async(_search)
