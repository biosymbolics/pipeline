# Point to this function as a handler in the Lambda configuration
import asyncio
from typing import Callable

from clients.low_level.prisma import get_prisma_client


def handle_async(func: Callable):
    def _handle(event, context):
        get_prisma_client(300)

        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(func(event, context))
        except Exception as e:
            return {"statusCode": 500, "body": f"Error: {e}"}

    return _handle