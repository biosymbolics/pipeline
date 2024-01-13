"""
Patent client
"""
import logging
import time
from prisma.models import BiomedicalEntity

from clients.low_level.prisma import prisma_context


from .types import AutocompleteResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


async def autocomplete(string: str, limit: int = 25) -> list[AutocompleteResult]:
    """
    Generates an autocomplete list for a given string and query

    Args:
        string (str): string to search for
        limit (int, optional): number of results to return. Defaults to 25.
    """
    start = time.monotonic()

    async with prisma_context(300):
        results = await BiomedicalEntity.prisma().find_many(
            where={"name": {"contains": string.lower()}},
            order={"count": "desc"},
            take=limit,
        )

    logger.info(
        "Autocomplete for string %s took %s seconds (%s)",
        string,
        round(time.monotonic() - start, 2),
        len(results),
    )

    return [
        AutocompleteResult(id=entity.name, label=f"{entity.name} ({entity.count})")
        for entity in results
    ]
