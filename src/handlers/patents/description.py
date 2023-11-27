from typing import TypedDict
from typing_extensions import NotRequired
import logging

from clients.low_level.boto3 import retrieve_with_cache_check
from clients.openai.gpt_client import GptApiClient
from utils.string import get_id


class DescribeParams(TypedDict):
    skip_cache: NotRequired[bool]
    terms: str


class DescribeEvent(TypedDict):
    queryStringParameters: DescribeParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def describe(event: DescribeEvent, context):
    """
    Get GPT description of terms

    Invocation:
    - Local: `serverless invoke local --function describe-terms --data='{"queryStringParameters": { "terms":"asthma;melanoma", "skip_cache": true }}'`
    - Remote: `serverless invoke --function describe-terms --data='{"queryStringParameters": { "terms":"gpr84", "skip_cache": true }}'`
    - API: `curl https://api.biosymbolics.ai/terms/describe?terms=gpr84`
    """
    gpt_client = GptApiClient(model="gpt-3.5-turbo")  # gpt-4 too slow

    params = event.get("queryStringParameters", {})
    terms = params.get("terms")
    terms_list = terms.split(";") if terms else []
    skip_cache = params.get("skip_cache", False)

    if not params or not terms or not all([len(t) > 1 for t in terms_list]):
        logger.error(
            "Missing or malformed query params: %s",
            params,
        )
        return {"statusCode": 400, "body": "Missing parameter(s)"}

    logger.info(
        "Fetching description for terms: %s",
        terms_list,
    )

    if skip_cache:
        description = gpt_client.describe_terms(terms_list)
    else:
        key = f"gpt-description-{get_id(terms_list)}"
        description = retrieve_with_cache_check(
            lambda: gpt_client.describe_terms(terms_list), key=key
        )

    return {"statusCode": 200, "body": description}
