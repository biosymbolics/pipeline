"""
Patent client
"""

import logging
import time

from prisma import Prisma


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

    results = await Prisma().query_raw(
        f"""
        SELECT name, count from (
            SELECT name, count
            FROM biomedical_entity
            WHERE search @@ plainto_tsquery('english', $1)

            UNION

            SELECT name, count
            FROM owner
            WHERE search @@ plainto_tsquery('english', $1)
        ) s ORDER BY count DESC
        LIMIT {limit}
        """,
        string,
    )

    logger.info(
        "Autocomplete for string %s took %s seconds (%s)",
        string,
        round(time.monotonic() - start, 2),
        len(results),
    )

    return [
        AutocompleteResult(
            id=result["name"], label=f"{result['name']} ({result['count']})"
        )
        for result in results
    ]
