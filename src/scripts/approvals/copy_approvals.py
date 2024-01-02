"""
Utils for copying approvals data
"""
from dataclasses import dataclass
from datetime import datetime
import re
import sys
from pydash import flatten, uniq
from prisma import Prisma
from prisma.models import (
    BiomedicalEntity,
    Indicatable,
    Intervenable,
    RegulatoryApproval,
)
from prisma.types import BiomedicalEntityCreateInput
from prisma.enums import BiomedicalEntityType
import asyncio
import logging

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from core.ner import TermNormalizer
from constants.patterns.intervention import PRIMARY_MECHANISM_BASE_TERMS
from constants.core import REGULATORY_APPROVAL_TABLE
from typings.core import Dataclass
from utils.re import get_or_re

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SOURCE_DB = "drugcentral"
DEST_DB = "patents"
SOURCE_FIELDS = [
    "MAX(prod.ndc_product_code) as id",
    "prod.product_name as brand_name",
    "(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) as generic_name",
    "ARRAY_AGG(distinct struct.name)::text[] as active_ingredients",
    # "MAX(prod_approval.applicant) as applicant",
    "MAX(prod.marketing_status) as application_type",
    "MAX(approval.approval) as approval_date",
    "MAX(approval.type) as agency",
    "array_remove(ARRAY_AGG(distinct metadata.concept_name), NULL) as indications",
    "JSON_AGG(pharma_class.*) as pharmacologic_classes",
    "MAX(label.pdf_url) as url",
]
SUPPRESSION_APPROVAL_TYPES = tuple(
    [
        "otc monograph not final",
        "otc monograph final",
        "unapproved drug other",
    ]
)


def get_preferred_pharmacologic_class(pharmacologic_classes: list[dict]) -> str | None:
    """
    Temporary/hack solution for getting the preferred pharmacologic class
    """

    def get_priority(pc: dict) -> int:
        score = 0
        if (
            re.match(
                f".*{get_or_re(list(PRIMARY_MECHANISM_BASE_TERMS.values()))}.*",
                pc["name"],
                flags=re.IGNORECASE,
            )
            is not None
        ):
            score += 10
        if pc["type"] == "MoA":
            score += 3
        elif pc["type"] == "EPC":
            score += 2
        elif pc["type"] == "MESH":
            score += 1

        return score

    prioritized = sorted(
        [pa for pa in pharmacologic_classes if (pa or {}).get("name") is not None],
        key=get_priority,
        reverse=True,
    )
    if len(prioritized) == 0:
        return None
    return prioritized[0]["name"].lower()


def get_source_sql(fields=SOURCE_FIELDS):
    return f"""
    SELECT {", ".join(fields)}
    FROM
        approval,
        active_ingredient,
        product prod,
        prd2label p2l,
        label,
        structures struct
    LEFT JOIN pharma_class on pharma_class.struct_id = struct.id
    LEFT JOIN omop_relationship metadata on metadata.struct_id = struct.id and metadata.relationship_name = 'indication'
    WHERE approval.struct_id = struct.id -- TODO: combo drugs??
    AND active_ingredient.struct_id = struct.id
    AND active_ingredient.ndc_product_code = prod.ndc_product_code
    AND p2l.ndc_product_code = prod.ndc_product_code
    AND p2l.label_id = label.id
    AND lower(prod.marketing_status) not in {SUPPRESSION_APPROVAL_TYPES}
    AND approval.approval is not null
    AND prod.product_name is not null
    GROUP BY prod.product_name
"""


@dataclass(frozen=True)
class InterventionIntermediate(Dataclass):
    generic_name: str
    brand_name: str
    active_ingredients: list[str]
    pharmacologic_classes: list[str]


async def copy_interventions():
    normalizer = TermNormalizer()
    fields = [
        "lower(prod.product_name) as brand_name",
        "lower(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) as generic_name",
        "ARRAY_AGG(distinct lower(struct.name))::text[] as active_ingredients",
        "ARRAY_REMOVE(ARRAY_AGG(distinct lower(pharma_class.name)), NULL)::text[] as pharmacologic_classes",
    ]
    records = PsqlDatabaseClient(SOURCE_DB).select(query=get_source_sql(fields))
    iis = [InterventionIntermediate(**r) for r in records]

    db = Prisma(auto_register=True)
    await db.connect()

    constituents = flatten(
        [r["active_ingredients"] for r in records if len(r["active_ingredients"]) > 1]
    )
    pharma_classes = flatten([r["pharmacologic_classes"] for r in records])
    generic_names = flatten([r["generic_name"] for r in records])

    # slow
    all_ints = uniq([*generic_names, *constituents, *pharma_classes])
    linked_ents = normalizer.normalize(all_ints)

    # TODO: link only? or link only and if no match, NER?
    link_map = {k: v.linked_entity for k, v in zip(all_ints, linked_ents)}

    await BiomedicalEntity.prisma().create_many(
        data=[
            *[
                {"name": c, "entity_type": BiomedicalEntityType.COMPOUND}
                for c in uniq(flatten(constituents))
            ],
            *[
                {"name": pc, "entity_type": BiomedicalEntityType.MECHANISM}
                for pc in uniq(flatten(pharma_classes))
            ],
        ],
        skip_duplicates=True,
    )

    for ii in iis:
        crud: BiomedicalEntityCreateInput = {
            "name": ii.generic_name,
            "comprised_of": {
                "connect": [{"name": i} for i in ii.active_ingredients],
            },
            "parents": {
                "connect": [{"name": i} for i in ii.pharmacologic_classes],
            },
            "entity_type": BiomedicalEntityType.COMPOUND,
            "synonyms": [ii.brand_name],
        }
        await BiomedicalEntity.prisma().upsert(
            where={"name": ii.generic_name},
            data={
                "create": crud,
                "update": crud,  # type: ignore
            },
        )


async def copy_all_approvals():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)
    """
    approvals = PsqlDatabaseClient(SOURCE_DB).select(query=get_source_sql())
    db = Prisma(auto_register=True)
    await db.connect()

    # create approval records
    await RegulatoryApproval.prisma().create_many(
        data=[
            {
                "id": a["id"],
                "agency": a["agency"],
                "approval_date": datetime(*a["approval_date"].timetuple()[:6]),
                "application_type": a["application_type"],
                "url": a["url"],
                "text_for_search": " ".join(
                    [
                        a["brand_name"],
                        *a["active_ingredients"],
                        *a["indications"],
                        *[
                            _a["name"]
                            for _a in a["pharmacologic_classes"]
                            if _a is not None and _a["name"] is not None
                        ],
                    ]
                ),
            }
            for a in approvals
        ],
        skip_duplicates=True,
    )

    await Indicatable.prisma().create_many(
        data=[
            {"name": i, "regulatory_approval_id": a["id"]}
            for a in approvals
            for i in a["indications"]
        ],
        skip_duplicates=True,
    )

    await Intervenable.prisma().create_many(
        data=[
            {
                "name": a["generic_name"] or a["brand_name"],
                "instance_rollup": get_preferred_pharmacologic_class(
                    a["pharmacologic_classes"]
                ),
                "is_primary": True,
                "regulatory_approval_id": a["id"],
            }
            for a in approvals
        ],
        skip_duplicates=True,
    )

    await db.disconnect()
    raw_client = PsqlDatabaseClient()
    raw_client.execute_query(
        f"""
        UPDATE {REGULATORY_APPROVAL_TABLE} SET search = to_tsvector('english', text_for_search)
        """,
    )
    raw_client.create_indices(
        [
            {
                "table": REGULATORY_APPROVAL_TABLE,
                "column": "search",
                "is_gin": True,
            },
        ]
    )


async def copy_approvals():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)
    """
    # await copy_all_approvals()
    await copy_interventions()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
        Usage: python3 -m scripts.approvals.copy_approvals
        Copies approvals data to postgres

        update trials set pharmacologic_class=a.pharmacologic_class from regulatory_approvals a where lower(a.generic_name)=lower(trials.intervention);
        update annotations set instance_rollup=a.pharmacologic_class from regulatory_approvals a where lower(term)=lower(a.generic_name);
        """
        )
        sys.exit()

    asyncio.run(copy_approvals())
