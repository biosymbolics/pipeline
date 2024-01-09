"""
Patent client
"""
import logging
import time
from prisma.models import BiomedicalEntity


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

    results = await BiomedicalEntity.prisma().find_many(
        where={"name": {"contains": string}},
        take=limit,
    )

    logger.info(
        "Autocomplete for string %s took %s seconds",
        string,
        round(time.monotonic() - start, 2),
    )

    return [{"id": entity.id, "label": entity.name} for entity in results]
