import json
from typing import TypedDict

from clients import patents as patent_client


AutocompleteParams = TypedDict("AutocompleteParams", {"term": str})
AutocompleteEvent = TypedDict(
    "AutocompleteEvent", {"queryStringParameters": AutocompleteParams}
)


def autocomplete(event: AutocompleteEvent, context):
    """
    Autocomplete terms
    """
    params = event["queryStringParameters"]
    term = params.get("term")

    terms = patent_client.autocomplete_terms(term)

    return {"statusCode": 200, "body": json.dumps(terms)}
