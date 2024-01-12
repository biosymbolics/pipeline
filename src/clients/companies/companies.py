from typing import Sequence
from prisma.models import Owner, Ownable
from prisma.types import OwnableInclude, OwnableWhereInput, StringFilter
from functools import lru_cache

from pydash import uniq_by
from clients.low_level.prisma import prisma_context

from typings.companies import CompanyFinancials


ID_FIELDS = {k: v for k, v in OwnableWhereInput.__annotations__.items() if "id" in k}


async def get_owner_map(
    ids: Sequence[str], id_field: str = "patent_id"
) -> dict[str, Owner]:
    """
    Fetch companies matching names.
    Return a map between owner name and Owner object.
    Cached.

    Args:
        ids: ids for which to fetch associated owners
        id_field: field to use for id matching (e.g. "patent_id")
    """
    if id_field not in ID_FIELDS:
        raise ValueError(f"Invalid id_field: {id_field}; must be one of {ID_FIELDS}")

    where = {id_field: StringFilter({"in": list(ids)})}

    ownables = await Ownable.prisma().find_many(
        where=OwnableWhereInput(**where),  # type: ignore
        include=OwnableInclude(owner={"include": {"financial_snapshot": True}}),
    )

    owners = uniq_by([o.owner for o in ownables if o.owner is not None], lambda o: o.id)

    return {owner.name: owner for owner in owners}


async def get_financial_map(
    ids: Sequence[str], id_field: str = "patent_id"
) -> dict[str, CompanyFinancials]:
    """
    Fetch companies matching names.
    Return a map between company name and Company object.
    Cached.

    Args:
        ids: ids for which to fetch associated owners
        id_field: field to use for id matching (e.g. "patent_id")
    """
    owner_map = await get_owner_map(tuple(ids), id_field)
    return {
        owner.name: CompanyFinancials(**owner.financial_snapshot.model_dump())
        for owner in owner_map.values()
        if owner.financial_snapshot is not None
    }
