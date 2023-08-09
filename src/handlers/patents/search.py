import json
from typing_extensions import NotRequired
from typing import TypedDict

from clients import patents as patent_client
from clients.patents import RelevancyThreshold


class SearchParams(TypedDict):
    terms: list[str]
    fetch_approval: NotRequired[bool]
    min_patent_years: NotRequired[int]
    relevancy_threshold: NotRequired[RelevancyThreshold]
    max_results: NotRequired[int]


class SearchEvent(TypedDict):
    queryStringParameters: SearchParams


def search(event: SearchEvent, context):
    """
    Search patents by terms
    """
    params = event["queryStringParameters"]
    terms = params.get("terms")
    fetch_approval = params.get("fetch_approval") or False
    min_patent_years = params.get("min_patent_years") or 10
    relevancy_threshold = params.get("relevancy_threshold") or "high"
    max_results = params.get("max_results") or 100

    patents = patent_client.search(
        terms, fetch_approval, min_patent_years, relevancy_threshold, max_results
    )

    return {"statusCode": 200, "body": json.dumps(patents)}
