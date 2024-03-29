from typing import TypedDict
from typing_extensions import NotRequired
import logging

from clients.llm.llm_client import GptApiClient
from handlers.utils import handle_async


class DescribeParams(TypedDict):
    skip_cache: NotRequired[bool]
    terms: str


class DescribeEvent(TypedDict):
    queryStringParameters: DescribeParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _describe(event: DescribeEvent, context):
    """
    Get GPT description of terms

    Invocation:
    - Local: `serverless invoke local --function describe-terms --data='{"queryStringParameters": { "terms":"asthma;melanoma", "skip_cache": true }}'`
    - Remote: `serverless invoke --function describe-terms --data='{"queryStringParameters": { "terms":"gpr84", "skip_cache": true }}'`
    - API: `curl https://api.biosymbolics.ai/terms/describe?terms=gpr84`
    """

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

    gpt_client = GptApiClient(model="gpt-3.5-turbo", skip_cache=skip_cache)

    logger.info(
        "Fetching description for terms: %s",
        terms_list,
    )

    description = await gpt_client.describe_terms(terms_list)

    return {"statusCode": 200, "body": description}


describe = handle_async(_describe)
