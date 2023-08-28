import json
from typing import TypedDict
import logging
from urllib.parse import unquote

from clients.sec.chat import SecChatClient


class SecChatParams(TypedDict):
    question: str


class SecChatEvent(TypedDict):
    queryStringParameters: SecChatParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def chat(event: SecChatEvent, context):
    """
    Query SEC docs

    Invocation:
    - Local: `serverless invoke local --function ask-sec --data='{"queryStringParameters": { "question": "What drugs were in the Biogen pipeline in 2023?" }}'`
    - Remote: `serverless invoke --function ask-sec --data='{"queryStringParameters": { "question": "What drugs were in the Biogen pipeline in 2023?" }}'`
    - API: `curl 'https://api.biosymbolics.ai/sec/ask?question=What%20drugs%20were%20in%20the%20Biogen%20pipeline%20in%202023?'`
    """
    sec_chat = SecChatClient()

    params = event.get("queryStringParameters", {})
    question = unquote(params["question"]) if params.get("question") else None

    if not question or len(question) < 5:
        logger.error(
            "Missing or malformed query params: %s",
            params,
        )
        return {
            "statusCode": 400,
            "body": "Missing parameter(s)",
        }

    logger.info(
        "Fetching answer for question: %s",
        question,
    )

    answer = sec_chat.ask_question(question)

    return {"statusCode": 200, "body": json.dumps(answer)}
