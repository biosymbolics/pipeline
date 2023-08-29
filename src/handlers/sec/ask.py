from datetime import datetime
import json
from typing import TypedDict
from typing_extensions import NotRequired
import logging
from urllib.parse import unquote

from clients.sec.ask import AskSecClient


class SecChatParams(TypedDict):
    question: str
    question_type: NotRequired[str]


class SecChatEvent(TypedDict):
    queryStringParameters: SecChatParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def ask(event: SecChatEvent, context):
    """
    Query SEC docs

    Invocation:
    - Local: `serverless invoke local --function ask-sec --data='{"queryStringParameters": { "question": "What drugs were in the Biogen pipeline in 2023?" }}'`
    - Remote: `serverless invoke --function ask-sec --data='{"queryStringParameters": { "question": "Tell me about anti-thrombin antibodies" }}'`
    - API: `curl 'https://api.biosymbolics.ai/sec/ask?question=What%20drugs%20were%20in%20the%20Biogen%20pipeline%20in%202023?'`
    """
    ask_sec = AskSecClient()

    params = event.get("queryStringParameters", {})
    question = unquote(params["question"]) if params.get("question") else None
    question_type = params.get("question_type") or "source"

    if not question or len(question) < 3:
        logger.error(
            "Missing or malformed query params: %s",
            params,
        )
        return {
            "statusCode": 400,
            "body": "Missing parameter(s)",
        }

    logger.info("Fetching answer for question: %s (%s)", question, question_type)

    if question_type == "entity":
        answer = ask_sec.ask_about_entity(question)
    elif question_type == "events":
        answer = json.dumps(ask_sec.get_events(question, datetime(2020, 1, 1)))
    else:
        answer = ask_sec.ask_question(question)

    return {"statusCode": 200, "body": answer}
