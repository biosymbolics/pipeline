"""
Patent client
"""

import logging
import time
from typing import Sequence


from clients.low_level.prisma import prisma_context
from typings.client import AutocompleteParams

from .types import AutocompleteResult


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TYPE_TABLE_MAP = {"entity": "biomedical_entity", "owner": "owner"}


async def autocomplete(p: AutocompleteParams) -> list[AutocompleteResult]:
    """
    Generates an autocomplete list for a given string and query

    Args:
        string (str): string to search for
        limit (int, optional): number of results to return. Defaults to 25.
    """
    start = time.monotonic()

    queries = [
        f"""
        SELECT name, count
        FROM {TYPE_TABLE_MAP[type]}
        WHERE search @@ plainto_tsquery('english', $1)
        """
        for type in p.types
    ]
    query = " UNION ".join(queries)

    async with prisma_context(300) as db:
        results = await db.query_raw(
            f"""
            SELECT name, count FROM ({query}) s
            ORDER BY count DESC LIMIT {p.limit}
            """,
            p.string,
        )

    logger.info(
        "Autocomplete for string %s took %s seconds (%s)",
        p.string,
        round(time.monotonic() - start, 2),
        len(results),
    )

    return [
        AutocompleteResult(
            id=result["name"], label=f"{result['name']} ({result['count']})"
        )
        for result in results
    ]
