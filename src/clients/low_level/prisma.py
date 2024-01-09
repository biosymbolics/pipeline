from prisma.client import Prisma, register
import logging
import os

from constants.core import DATABASE_URL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

PRISMA_CLIENT = None
os.environ["DATABASE_URL"] = DATABASE_URL or ""


def get_prisma_client(timeout: int | None) -> Prisma:
    """
    Get a Prisma client

    Centralized with global to avoid re-registering the client (which would fail)
    https://prisma-client-py.readthedocs.io/en/stable/reference/model-actions/
    """
    global PRISMA_CLIENT

    if PRISMA_CLIENT is None:
        logger.info("Registering Prisma client")
        print("Registering Prisma client")
        client = Prisma(http={"timeout": timeout})
        PRISMA_CLIENT = client
    else:
        client = PRISMA_CLIENT

    if not client.is_registered():
        logger.debug("Prisma client is not registered")
        print("Prisma client is not registered")
        register(client)

    # if not client.is_connected():
    #     print("CONNECTING CLIENT")
    #     await client.connect()

    return client
