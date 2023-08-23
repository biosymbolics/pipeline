from typing import TypedDict
import logging

from clients.sec.chat import SecChatClient


class ClinDevParams(TypedDict):
    indication: str


class ClinDevEvent(TypedDict):
    queryStringParameters: ClinDevParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def query_clindev(event: ClinDevEvent, context):
    """
    Query GPT about clindev timelines

    Invocation:
    - Local: `serverless invoke local --function query-clindev --data='{"queryStringParameters": { "indication": "asthma" }}'`
    - Remote: `serverless invoke --function query-clindev --data='{"queryStringParameters": { "indication": "asthma" }}'`
    - API: `curl https://api.biosymbolics.ai/sec/query/clindev?indication=asthma`
    """
    sec_chat = SecChatClient()

    params = event.get("queryStringParameters", {})
    indication = params.get("indication")

    if not indication or len(indication) < 5:
        logger.error(
            "Missing or malformed query params: %s",
            params,
        )
        return {
            "statusCode": 400,
            "body": "Missing parameter(s)",
        }

    logger.info(
        "Fetching info for indication: %s",
        indication,
    )

    answer = sec_chat.query_clindev(indication)

    return {"statusCode": 200, "body": answer}
