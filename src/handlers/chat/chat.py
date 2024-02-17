import json
from typing import TypedDict
import logging
from pydantic import BaseModel
from prisma.enums import MockChatType

from clients.biosym_chat.mock_chat import MockChatClient
from clients.low_level.boto3 import retrieve_with_cache_check
from clients.openai.gpt_client import GptApiClient
from handlers.utils import handle_async
from typings.core import ResultBase
from utils.encoding.json_encoder import DataclassJSONEncoder


class ChatParams(BaseModel):
    skip_cache: bool = False
    prompt: str
    conversation_key: str | None = None
    message_key: str | None = None


class ChatEvent(TypedDict):
    queryStringParameters: ChatParams


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class ChatResponse(ResultBase):
    content: str
    type: MockChatType | None = None


async def _chat(raw_event: dict, context):
    """
    Get GPT description of terms

    Invocation:
    - Local: `serverless invoke local --function chat --data='{"queryStringParameters": { "prompt":"tell me about asthma and melanoma", "skip_cache": true }}'`
    - Remote: `serverless invoke --function chat --data='{"queryStringParameters": { "prompt":"tell me about gpr84", "skip_cache": true }}'`
    - API: `curl https://api.biosymbolics.ai/chat?prompt=tell me about gpr84`
    """
    gpt_client = GptApiClient(model="gpt-4")  # gpt-3.5-turbo

    p = ChatParams(**(raw_event["queryStringParameters"] or {}))

    logger.info("Asking llm: %s", p.prompt)

    if p.conversation_key is not None and p.message_key is not None:
        response = await MockChatClient(conversation_key=p.conversation_key).query(
            p.message_key
        )
        if response is not None:
            return {
                "statusCode": 200,
                "body": json.dumps(
                    ChatResponse(**response.__dict__), cls=DataclassJSONEncoder
                ),
            }

    if p.skip_cache:
        answer = gpt_client.query(p.prompt)
    else:
        key = f"gpt-chat-{p.prompt}"
        answer = await retrieve_with_cache_check(
            lambda: gpt_client.query(p.prompt), key=key
        )

    return {
        "statusCode": 200,
        "body": json.dumps(ChatResponse(content=answer), cls=DataclassJSONEncoder),
    }


chat = handle_async(_chat)
