"""
Utils for copying approvals data
"""
from datetime import datetime
import re
import sys
from prisma import Prisma
from prisma.models import (
    BiomedicalEntity,
    Indicatable,
    Intervenable,
    RegulatoryApproval,
)
from prisma.enums import BiomedicalEntityType, Source
from prisma.types import (
    BiomedicalEntityCreateInput,
    BiomedicalEntityCreateWithoutRelationsInput,
)
import asyncio
import logging
from pydash import compact, group_by, omit, uniq

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from core.ner import TermNormalizer
from core.ner.types import CanonicalEntity
from constants.patterns.intervention import PRIMARY_MECHANISM_BASE_TERMS
from constants.core import REGULATORY_APPROVAL_TABLE
from utils.re import get_or_re

from .types import InterventionIntermediate

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


def merge_insert_records(
    groups: list[BiomedicalEntityCreateInput],
) -> BiomedicalEntityCreateInput:
    """
    Merge records with same canonical id
    """
    return {
        "canonical_id": groups[0].get("canonical_id"),
        "name": groups[0]["name"],
        "entity_type": groups[0]["entity_type"],
        "synonyms": uniq(compact([s for g in groups for s in g.get("synonyms") or []])),
        "sources": groups[0].get("sources") or [],
    }


def create_intervention_records(
    iis: list[InterventionIntermediate],
    canonical_map: dict[str, CanonicalEntity],
    intervention_type_map: dict[str, BiomedicalEntityType],
) -> list[BiomedicalEntityCreateInput]:
    """
    Create records for intervention insert
    """

    def get_insert_record(ii: InterventionIntermediate) -> BiomedicalEntityCreateInput:
        canonical = canonical_map.get(ii.generic_name)

        active_ingredient_ids = [
            canonical_map[i].id if id in canonical_map else None
            for i in ii.active_ingredients
        ]

        pharmacologic_class_ids = [
            canonical_map[i].id if i in canonical_map else None
            for i in ii.pharmacologic_classes
        ]

        if canonical is not None:
            conditional_fields = {
                "canonical_id": canonical.id,
                "name": canonical.name.lower(),
                "entity_type": canonical.type,
                "sources": [Source.UMLS],
            }
        else:
            conditional_fields = {
                "canonical_id": None,
                "name": ii.generic_name,
                "entity_type": intervention_type_map[ii.generic_name],
                "sources": [Source.FDA],
            }

        return BiomedicalEntityCreateInput(
            **{
                **conditional_fields,  # type: ignore
                "synonyms": [ii.generic_name, ii.brand_name],  # TODO: include UMLS syns
                "comprised_of": {
                    "connect": [
                        {"canonical_id": ai_id}
                        for ai_id in uniq(compact(active_ingredient_ids))
                    ],
                },
                "parents": {
                    "connect": [
                        {"canonical_id": pc_id}
                        for pc_id in uniq(compact(pharmacologic_class_ids))
                    ],
                },
            }
        )

    # merge records with same canonical id
    irs = [get_insert_record(ii) for ii in iis]
    irs_grouped = group_by(
        [ir for ir in irs if ir.get("canonical_id") is not None], "canonical_id"
    )
    all_intervention_records = [
        *[
            merge_insert_records(groups)
            for id, groups in irs_grouped.items()
            if id is not None
        ],
        *[ir for ir in irs if ir.get("canonical_id") is None],
    ]
    return all_intervention_records


def get_intervention_canonical_map(
    intervention_type_map: dict[str, BiomedicalEntityType], normalizer: TermNormalizer
):
    """
    Get canonical map for interventions
    """
    non_combination_type_map = {
        k: v
        for k, v in intervention_type_map.items()
        if v != BiomedicalEntityType.COMBINATION
    }

    # normalize all intervention names, except if combos
    canonical_docs = normalizer.normalize_strings(list(non_combination_type_map.keys()))

    # map for quick lookup of links
    canonical_map = {
        nt[0]: de.canonical_entity
        for nt, de in zip(non_combination_type_map.items(), canonical_docs)
        if de.canonical_entity is not None
    }
    return canonical_map


async def copy_interventions():
    normalizer = TermNormalizer(candidate_selector="CandidateSelector")
    fields = [
        "lower(prod.product_name) as brand_name",
        "lower(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) as generic_name",
        "ARRAY_AGG(distinct lower(struct.name))::text[] as active_ingredients",
        "ARRAY_REMOVE(ARRAY_AGG(distinct lower(pharma_class.name)), NULL)::text[] as pharmacologic_classes",
    ]
    source_records = PsqlDatabaseClient(SOURCE_DB).select(query=get_source_sql(fields))
    source_interventions = [InterventionIntermediate(**r) for r in source_records]

    db = Prisma(auto_register=True)
    await db.connect()

    intervention_type_map = {
        # drugs broken out by combo (more than one active ingredient) or single/compound
        **{
            i.generic_name: BiomedicalEntityType.COMBINATION
            if len(i.active_ingredients) > 1
            else BiomedicalEntityType.COMPOUND
            for i in source_interventions
        },
        # active ingredients for combination drugs
        **{
            ai: BiomedicalEntityType.COMPOUND
            for i in source_interventions
            for ai in i.active_ingredients
            if len(i.active_ingredients) > 1
        },
        # mechanisms / pharmacologic classes
        **{
            pc: BiomedicalEntityType.MECHANISM
            for i in source_interventions
            for pc in i.pharmacologic_classes
        },
    }

    canonical_map = get_intervention_canonical_map(intervention_type_map, normalizer)
    intervention_recs = create_intervention_records(
        source_interventions, canonical_map, intervention_type_map
    )

    # create flat records
    await BiomedicalEntity.prisma().create_many(
        data=[
            BiomedicalEntityCreateWithoutRelationsInput(
                **omit(dict(ir), "comprised_of", "parents")  # type: ignore
            )
            for ir in intervention_recs
        ],
        skip_duplicates=True,
    )

    # update records with relationships
    intervention_recs_nested = [
        ir
        for ir in intervention_recs
        if (ir.get("comprised_of") or ir.get("parents")) is not None
    ]
    for irn in intervention_recs_nested:
        update = {
            "comprised_of": irn.get("comprised_of"),
            "parents": irn.get("parents"),
        }
        await BiomedicalEntity.prisma().update(
            where={"name": irn["name"]}, data=update  # type: ignore
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

    # create "indicatable" records, those that map approval to a canonical indication
    await Indicatable.prisma().create_many(
        data=[
            {"name": i, "regulatory_approval_id": a["id"]}
            for a in approvals
            for i in a["indications"]
        ],
        skip_duplicates=True,
    )

    # create "intervenable" records, those that map approval to a canonical intervention
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

    # create search index (unsupported by Prisma)
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
