from prisma.models import MockChat

from clients.low_level.prisma import prisma_context


class MockChatClient:
    """
    Mock chat client for demo purposes
    """

    def __init__(self, conversation_key: str, delay: int = 0):
        self.conversation_key = conversation_key
        self.delay = delay

    async def query(self, message_key: str) -> MockChat | None:
        """
        Get a message by its key
        """
        async with prisma_context(300) as db:
            message = await MockChat.prisma(db).find_unique(
                where={
                    "conversation_id_message_id": {
                        "conversation_id": self.conversation_key,
                        "message_id": message_key,
                    }
                },
            )

        return message
