from prisma.client import Prisma, register
import logging
import os

from constants.core import DATABASE_URL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PRISMA_CLIENT = None

if DATABASE_URL is None:
    raise ValueError("DATABASE_URL is not set")

os.environ["DATABASE_URL"] = DATABASE_URL


async def prisma_client(timeout: int | None, connect: bool = True) -> Prisma:
    client = prisma_context(timeout)
    if connect and not client.is_connected():
        await client.connect()
    return client


def prisma_context(timeout: int | None) -> Prisma:
    """
    Get a Prisma client

    Centralized with global to avoid re-registering the client (which would fail)
    https://prisma-client-py.readthedocs.io/en/stable/reference/model-actions/
    """
    global PRISMA_CLIENT

    if PRISMA_CLIENT is None:
        logger.info("Registering Prisma client")
        client = Prisma(http={"timeout": timeout})
        PRISMA_CLIENT = client
    else:
        client = PRISMA_CLIENT

    if not client.is_registered():
        logger.debug("Registering prisma client")
        register(client)

    return client
