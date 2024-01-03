"""
Utils for copying approvals data
"""
from dataclasses import dataclass
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
    BiomedicalEntityUpdateInput,
    BiomedicalEntityCreateWithoutRelationsInput,
)
import asyncio
import logging
from pydash import compact, flatten, group_by, omit, uniq

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from core.ner import TermNormalizer
from core.ner.types import CanonicalEntity
from constants.patterns.intervention import PRIMARY_MECHANISM_BASE_TERMS
from constants.core import REGULATORY_APPROVAL_TABLE
from utils.re import get_or_re

from .types import (
    BiomedicalEntityCreateInputWithRelationIds,
    InterventionIntermediate,
    RelationIdFieldMap,
)

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


def maybe_merge_insert_records(
    groups: list[BiomedicalEntityCreateInputWithRelationIds],
    canonical_id: str,
) -> list[BiomedicalEntityCreateInputWithRelationIds]:
    """
    Merge records with same canonical id
    """
    # if no canonical id, then no merging
    if canonical_id is None:
        return groups

    return [
        {
            "canonical_id": groups[0].get("canonical_id"),
            "name": groups[0]["name"],
            "entity_type": groups[0]["entity_type"],
            "synonyms": uniq(
                compact([s for g in groups for s in g.get("synonyms") or []])
            ),
            "sources": groups[0].get("sources") or [],
            "comprised_of": uniq(
                flatten([g.get("comprised_of") or [] for g in groups])
            ),
            "parents": uniq(flatten([g.get("parents") or [] for g in groups])),
        }
    ]


def create_entity_records(
    insert_map: dict[str, BiomedicalEntityType],
    source_map: dict[str, InterventionIntermediate],
    canonical_map: dict[str, CanonicalEntity],
    relation_id_map: RelationIdFieldMap,
    non_canonical_source: Source = Source.FDA,
    synonym_fields: list[str] = ["brand_name"],
) -> list[BiomedicalEntityCreateInputWithRelationIds]:
    """
    Create records for entity insert
    """

    def get_insert_record(orig_name: str) -> BiomedicalEntityCreateInputWithRelationIds:
        source_rec = source_map.get(orig_name)
        canonical = canonical_map.get(orig_name)

        if source_rec is not None:
            rel_fields: dict[str, list[str]] = {
                rel_field: uniq(
                    compact(
                        [
                            canonical_map[i].id if i in canonical_map else None
                            for i in source_rec[source_field]
                        ]
                    )
                )
                for rel_field, source_field in relation_id_map.items()
            }
            source_dependent_fields = {
                "synonyms": [
                    orig_name,
                    *[str(source_rec[sf]) for sf in synonym_fields],
                ],
                **rel_fields,
            }
        else:
            source_dependent_fields = {
                "synonyms": [orig_name],
            }

        if canonical is not None:
            canonical_dependent_fields = {
                "canonical_id": canonical.id,
                "name": canonical.name.lower(),
                "entity_type": canonical.type,
                "sources": [Source.UMLS],
            }
        else:
            canonical_dependent_fields = {
                "canonical_id": None,
                "name": orig_name,
                "entity_type": insert_map[orig_name],
                "sources": [non_canonical_source],
            }

        return BiomedicalEntityCreateInputWithRelationIds(
            **{
                **canonical_dependent_fields,  # type: ignore
                **source_dependent_fields,
            }
        )

    # merge records with same canonical id
    def merge_records():
        flat_recs = [get_insert_record(name) for name in insert_map.keys()]
        grouped_recs = group_by(flat_recs, "canonical_id")
        merged_recs = flatten(
            [
                maybe_merge_insert_records(groups, cid)
                for cid, groups in grouped_recs.items()
            ]
        )

        return merged_recs

    insert_records = merge_records()
    return insert_records


def get_canonical_map(
    entity_type_map: dict[str, BiomedicalEntityType], normalizer: TermNormalizer
):
    """
    Get canonical map for interventions
    """
    # normalize all intervention names, except if combos
    canonical_docs = normalizer.normalize_strings(list(entity_type_map.keys()))

    # map for quick lookup of canonical entities
    canonical_map = {
        nt[0]: de.canonical_entity
        for nt, de in zip(entity_type_map.items(), canonical_docs)
        if de.canonical_entity is not None
    }
    return canonical_map


async def copy_interventions():
    relation_id_map = RelationIdFieldMap(
        comprised_of="active_ingredients",
        parents="pharmacologic_classes",
    )
    normalizer = TermNormalizer(candidate_selector="CandidateSelector")
    fields = [
        "lower(prod.product_name) as brand_name",
        "lower(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) as generic_name",
        "ARRAY_AGG(distinct lower(struct.name))::text[] as active_ingredients",
        "ARRAY_REMOVE(ARRAY_AGG(distinct lower(pharma_class.name)), NULL)::text[] as pharmacologic_classes",
    ]
    source_records = PsqlDatabaseClient(SOURCE_DB).select(query=get_source_sql(fields))
    source_interventions = [InterventionIntermediate(**r) for r in source_records]
    source_map = {i.generic_name: i for i in source_interventions}

    db = Prisma(auto_register=True)
    await db.connect()

    insert_map = {
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

    normalize_type_map = {
        k: v for k, v in insert_map.items() if v != BiomedicalEntityType.COMBINATION
    }
    canonical_map = get_canonical_map(normalize_type_map, normalizer)
    entity_recs = create_entity_records(
        insert_map, source_map, canonical_map, relation_id_map
    )

    # create flat records
    await BiomedicalEntity.prisma().create_many(
        data=[
            BiomedicalEntityCreateWithoutRelationsInput(
                **omit(er, *relation_id_map.keys())  # type: ignore
            )
            for er in entity_recs
        ],
        skip_duplicates=True,
    )

    # update records with relationships with connection info
    recs_with_relations = [
        er
        for er in entity_recs
        if (any([er.get(k) is not None for k in relation_id_map.keys()])) is not None
    ]
    for rwr in recs_with_relations:
        update = BiomedicalEntityUpdateInput(
            **{  # type: ignore
                k: {"connect": [{"canonical_id": co} for co in rwr.get(k) or []]}
                for k in relation_id_map.keys()
            },
        )
        # print("updating", irn["name"], update)
        await BiomedicalEntity.prisma().update(where={"name": rwr["name"]}, data=update)


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
