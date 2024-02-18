import asyncio
import sys
from prisma.models import MockChat
from prisma.enums import MockChatType

from clients.low_level.prisma import prisma_client
from utils.file import load_json_from_file


async def load_mock_chat():
    """
    Load mock chat data
    """
    data = load_json_from_file("mock_chat.json")
    conversations = [
        {
            "conversation_id": key,
            "message_id": int(message["id"]),
            "content": message["content"],
            "description": message.get("description"),
            "type": message.get("type") or MockChatType.STANDARD,
        }
        for key, conversation in data.items()
        for message in conversation["messages"]
    ]
    client = await prisma_client(600)
    await MockChat.prisma(client).delete_many()
    await MockChat.prisma(client).create_many(conversations)  # type: ignore


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.misc.load_mock_chat
            UMLS etl
        """
        )
        sys.exit()

    asyncio.run(load_mock_chat())
