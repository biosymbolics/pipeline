from typing import TypedDict
from typing_extensions import NotRequired
import logging

from clients.low_level.boto3 import retrieve_with_cache_check
from clients.openai.gpt_client import GptApiClient
from handlers.utils import handle_async


class ChatParams(TypedDict):
    skip_cache: NotRequired[bool]
    prompt: str


class ChatEvent(TypedDict):
    queryStringParameters: ChatParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _chat(event: ChatEvent, context):
    """
    Get GPT description of terms

    Invocation:
    - Local: `serverless invoke local --function chat --data='{"queryStringParameters": { "prompt":"tell me about asthma and melanoma", "skip_cache": true }}'`
    - Remote: `serverless invoke --function chat --data='{"queryStringParameters": { "prompt":"tell me about gpr84", "skip_cache": true }}'`
    - API: `curl https://api.biosymbolics.ai/chat?prompt=tell me about gpr84`
    """
    gpt_client = GptApiClient(model="gpt-3.5-turbo")  # gpt-4 too slow

    params = event.get("queryStringParameters", {})
    prompt = params.get("prompt")
    skip_cache = params.get("skip_cache", False)

    if not params or not prompt:
        logger.error(
            "Missing or malformed query params: %s",
            params,
        )
        return {"statusCode": 400, "body": "Missing parameter(s)"}

    logger.info("Asking llm: %s", prompt)

    if skip_cache:
        answer = gpt_client.query(prompt)
    else:
        key = f"gpt-chat-{prompt}"
        answer = await retrieve_with_cache_check(
            lambda: gpt_client.query(prompt), key=key
        )

    return {"statusCode": 200, "body": answer}


chat = handle_async(_chat)
