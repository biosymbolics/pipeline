from typing import Awaitable, Callable, Sequence, TypeVar, TypedDict
from httpx import Limits
from prisma.client import Prisma, register
import logging
import os


from constants.core import DATABASE_URL
from typings.prisma import AllModelTypes
from utils.list import batch

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PRISMA_CLIENT = None

if DATABASE_URL is None:
    raise ValueError("DATABASE_URL is not set")

os.environ["DATABASE_URL"] = DATABASE_URL


async def prisma_client(timeout: int | None, do_connect: bool = True) -> Prisma:
    """
    Get a Prisma client
    """
    client = prisma_context(timeout)
    if do_connect and not client.is_connected():
        await client.connect()
    return client


def prisma_context(timeout: int | None) -> Prisma:
    """
    Get a Prisma context

    Centralized with global to avoid re-registering the client (which would fail)
    https://prisma-client-py.readthedocs.io/en/stable/reference/model-actions/
    """
    global PRISMA_CLIENT

    if PRISMA_CLIENT is None:
        logger.info("Registering Prisma client")
        client = Prisma(http={"limits": Limits(max_connections=50), "timeout": timeout})
        PRISMA_CLIENT = client
    else:
        client = PRISMA_CLIENT

    if not client.is_registered():
        try:
            logger.debug("Registering prisma client")
            register(client)
        except Exception as e:
            logger.debug("Error registering prisma client, perhaps a race condition")

    return client


T = TypeVar("T", bound=AllModelTypes)


async def batch_update(
    records: Sequence[T],
    update_func: Callable[[T, Prisma], Awaitable],
    batch_size: int = 1000,
    timeout: int | None = 300,
):
    """
    Batch update records (in a transaction, for performance)
    """
    client = await prisma_client(timeout)
    batches = batch(records, batch_size)
    for b in batches:
        async with client.tx() as tx:
            for r in b:
                try:
                    await update_func(r, tx)
                except Exception as e:
                    logger.warning("Error encountered in update: %s", e)
