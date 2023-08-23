from typing import TypedDict
import logging

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
    - Remote: `serverless invoke --function ask-sec --data='{"queryStringParameters": { "question": "What drugs were in Biogen's pipeline in 2023?" }}'`
    - API: `curl https://api.biosymbolics.ai/sec/ask?question='What drugs were in Biogen's pipeline in 2023?'`
    """
    sec_chat = SecChatClient()

    params = event.get("queryStringParameters", {})
    question = params.get("question")

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

    return {"statusCode": 200, "body": answer}
