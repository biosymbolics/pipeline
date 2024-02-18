from prisma.models import MockChat

from clients.low_level.prisma import prisma_context


class MockChatClient:
    """
    Mock chat client for demo purposes
    """

    def __init__(self, conversation_id: str, delay: int = 0):
        self.conversation_id = conversation_id
        self.delay = delay

    async def query(self, message_id: int) -> MockChat | None:
        """
        Get a message by its key
        """
        async with prisma_context(300) as db:
            message = await MockChat.prisma(db).find_unique(
                where={
                    "conversation_id_message_id": {
                        "conversation_id": self.conversation_id,
                        "message_id": message_id,
                    }
                },
            )

        return message
