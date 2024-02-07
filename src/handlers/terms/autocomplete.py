"""
Handler for terms autocomplete
"""

import json
import logging
from pydantic import BaseModel

from clients import terms as terms_client
from handlers.utils import handle_async

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class AutocompleteParams(BaseModel):
    string: str
    limit: int = 25


class AutocompleteEvent(BaseModel):
    queryStringParameters: AutocompleteParams


async def _autocomplete(raw_event: dict, context):
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

    terms = await terms_client.autocomplete(p.string, p.limit)

    return {"statusCode": 200, "body": json.dumps(terms)}


autocomplete = handle_async(_autocomplete)
