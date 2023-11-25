"""
Handler for patents autocomplete
"""
import json
import logging
from pydantic import BaseModel

from clients import patents as patent_client
from clients.patents.types import AutocompleteMode

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AutocompleteParams(BaseModel):
    string: str
    limit: int = 25
    mode: AutocompleteMode = "term"


class AutocompleteEvent(BaseModel):
    queryStringParameters: AutocompleteParams


def autocomplete(raw_event: dict, context):
    """
    Autocomplete terms or ids for patents

    Invocation:
    - Local: `serverless invoke local --function autocomplete --param='ENV=local' --data='{"queryStringParameters": { "string":"asthm" }}'`
    - Local: `serverless invoke local --function autocomplete --param='ENV=local' --data='{"queryStringParameters": { "string":"WO-0224", "mode": "id" }}'`
    - Remote: `serverless invoke --function autocomplete --data='{"queryStringParameters": { "string":"alzheim" }}'`
    - API: `curl https://api.biosymbolics.ai/autocomplete?string=asthm`

    Output (for string "asthm"):
    ```json
    {
        "statusCode": 200,
        "body": [
            "Allergic asthma (23702)",
            "Cardiac asthma (68)",
            "Cough variant asthma (703)",
            "Intrinsic asthma (1831)",
            "Late onset asthma (130)",
            "analgesic asthma syndrome (595)",
            "infantile asthma (114)"
        ]
    }
    ```
    """
    event = AutocompleteEvent(**raw_event)
    p = event.queryStringParameters

    if not p or not p.string:
        logger.error(
            "Missing query or param `string`, params: %s",
            p,
        )
        return {
            "statusCode": 400,
            "body": "Missing parameter(s)",
        }

    if len(p.string) < 3:
        logger.info("Term too short, skipping autocomplete")
        return {"statusCode": 200, "body": json.dumps([])}

    terms = patent_client.autocomplete(p.string, p.mode, p.limit)

    return {"statusCode": 200, "body": json.dumps(terms)}
