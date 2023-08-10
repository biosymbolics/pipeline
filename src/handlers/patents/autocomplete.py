"""
Handler for patents autocomplete
"""
import logging
from typing import TypedDict

from clients import patents as patent_client


AutocompleteParams = TypedDict("AutocompleteParams", {"term": str})
AutocompleteEvent = TypedDict(
    "AutocompleteEvent", {"queryStringParameters": AutocompleteParams}
)


def autocomplete(event: AutocompleteEvent, context):
    """
    Autocomplete term for patents (used in patent term autocomplete)

    Invocation:
    - Local: `serverless invoke local --function autocomplete-patents --data='{"queryStringParameters": { "term":"asthm" }}'`
    - Remote: `serverless invoke --function autocomplete-patents --data='{"queryStringParameters": { "term":"asthm" }}'`

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
    params = event.get("queryStringParameters", {})
    term = params.get("term")

    if not params or not term:
        logging.error(
            "Missing queryStringParameters or param `term`, params: %s",
            params,
        )
        return {
            "statusCode": 400,
            "message": "Missing parameter(s)",
        }

    terms = patent_client.autocomplete_terms(term)

    return {"statusCode": 200, "body": terms}
