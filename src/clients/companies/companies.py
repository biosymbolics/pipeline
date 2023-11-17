from functools import lru_cache
from clients.low_level.postgres.postgres import PsqlDatabaseClient
from constants.core import COMPANIES_TABLE_NAME
from typings.companies import Company


@lru_cache(maxsize=None)
def get_company_map() -> dict[str, Company]:
    """
    Fetch companies matching names.
    Return a map between company name and Company object.
    Cached.

    Args:
        names: company names to fetch
    """

    query = f"SELECT * FROM {COMPANIES_TABLE_NAME}"
    results = PsqlDatabaseClient().select(query)

    return {result["name"]: Company(**result) for result in results}
