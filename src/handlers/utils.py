import asyncio
import logging

from typing import Callable

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def handle_async(func: Callable):
    def _handle(event, context):
        logger.info("HI %s", event)
        try:
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(func(event, context))
        except Exception as e:
            return {"statusCode": 500, "body": f"Error: {e}"}

    return _handle
