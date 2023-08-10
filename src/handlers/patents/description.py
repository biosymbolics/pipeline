from typing import TypedDict
from clients.openai.gpt_client import GptApiClient
import logging


class DescribeParams(TypedDict):
    terms: str


class DescribeEvent(TypedDict):
    queryStringParameters: DescribeParams


logger = logging.getLogger(__name__)


def describe(event: DescribeEvent, context):
    """
    Get GPT description of terms

    Invocation:
    - Local: `serverless invoke local --function describe-terms --data='{"queryStringParameters": { "terms":"asthma,melanoma" }}'`
    - Remote: `serverless invoke --function describe-terms --data='{"queryStringParameters": { "terms":"asthma,melanoma" }}'`
    - API: `curl https://v8v4ij0xs4.execute-api.us-east-1.amazonaws.com/dev/terms/describe?terms=asthma`
    """
    gpt_client = GptApiClient(model="gpt-3.5-turbo")

    params = event.get("queryStringParameters", {})
    terms = params.get("terms")
    terms_list = terms.split(",") if terms else []

    if not params or not terms or not all([len(t) > 1 for t in terms_list]):
        logging.error(
            "Missing or malformed query params: %s",
            params,
        )
        return {
            "statusCode": 400,
            "message": "Missing parameter(s)",
        }

    logging.info(
        "Fetching description for terms: %s",
        terms_list,
    )

    description = gpt_client.describe_terms(terms_list)

    return {"statusCode": 200, "body": {"description": description}}
