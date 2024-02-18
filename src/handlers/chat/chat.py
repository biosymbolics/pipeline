import json
import time
from typing import TypedDict
import logging
import uuid
from pydantic import BaseModel
from prisma.models import MockChat
from prisma.partials import Chat

from clients.biosym_chat.mock_chat import MockChatClient
from clients.low_level.boto3 import retrieve_with_cache_check
from clients.openai.gpt_client import GptApiClient
from handlers.utils import handle_async
from typings.core import ResultBase
from utils.encoding.json_encoder import DataclassJSONEncoder


class ChatParams(BaseModel):
    skip_cache: bool = False
    content: str
    conversation_id: str | None = None
    message_id: int | None = None


class ChatEvent(TypedDict):
    queryStringParameters: ChatParams


class MockChatResponse(ResultBase, MockChat):
    pass


class ChatResponse(ResultBase, Chat):
    pass


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def _chat(raw_event: dict, context):
    """
    Get GPT description of terms

    Invocation:
    - Local: `serverless invoke local --function chat --data='{"queryStringParameters": { "content":"tell me about asthma and melanoma", "skip_cache": true }}'`
    - Remote: `serverless invoke --function chat --data='{"queryStringParameters": { "content":"tell me about gpr84", "skip_cache": true }}'`
    - API: `curl https://api.biosymbolics.ai/chat?content=tell me about gpr84`
    """
    gpt_client = GptApiClient(model="gpt-4")  # gpt-3.5-turbo

    p = ChatParams(**(raw_event["queryStringParameters"] or {}))

    logger.info("Asking llm: %s", p)

    if p.conversation_id is not None and p.message_id is not None:
        time.sleep(5)
        resp = await MockChatClient(conversation_id=p.conversation_id).query(
            p.message_id + 1
        )
        logger.info("Got response from mock chat: %s", resp)
        if resp is not None:
            return {
                "statusCode": 200,
                "body": json.dumps(
                    MockChatResponse(**resp.model_dump()),
                    cls=DataclassJSONEncoder,
                ),
            }

    if p.skip_cache:
        answer = gpt_client.query(p.content)
    else:
        key = f"gpt-chat-{p.content}"
        answer = await retrieve_with_cache_check(
            lambda: gpt_client.query(p.content), key=key
        )

    return {
        "statusCode": 200,
        "body": json.dumps(
            ChatResponse(
                message_id=0,
                conversation_id=p.conversation_id or str(uuid.uuid4()),
                content=answer,
            ),
            cls=DataclassJSONEncoder,
        ),
    }


chat = handle_async(_chat)
