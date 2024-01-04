"""
Regulatory approval ETL script
"""
from datetime import datetime
import re
import sys
import asyncio
import logging
from pydash import compact
from prisma.enums import BiomedicalEntityType
from prisma.models import (
    Indicatable,
    Intervenable,
    RegulatoryApproval,
)


from clients.low_level.postgres import PsqlDatabaseClient
from constants.patterns.intervention import PRIMARY_MECHANISM_BASE_TERMS
from data.etl.biomedical_entity import BiomedicalEntityEtl
from data.etl.document import DocumentEtl
from data.etl.types import RelationConnectInfo, RelationIdFieldMap
from utils.re import get_or_re

from .types import InterventionIntermediate

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


SOURCE_DB = "drugcentral"
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


class ApprovalEtl(DocumentEtl):
    @staticmethod
    def get_source_sql(fields: list[str]):
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

    async def copy_indications(self):
        """
        Create indication records
        """
        fields = [
            "array_remove(ARRAY_AGG(distinct lower(metadata.concept_name)), NULL) as indications"
        ]
        source_records = PsqlDatabaseClient(SOURCE_DB).select(
            query=ApprovalEtl.get_source_sql(fields)
        )

        source_map = {
            i: {"synonyms": [i], "default_type": BiomedicalEntityType.DISEASE}
            for sr in source_records
            for i in sr["indications"]
        }

        terms_to_insert = list(source_map.keys())
        terms_to_canonicalize = terms_to_insert

        await BiomedicalEntityEtl(
            "CandidateSelector",
            relation_id_field_map=RelationIdFieldMap(
                synonyms=RelationConnectInfo(
                    source_field="synonyms", dest_field="term", input_type="create"
                ),
            ),
        ).create_records(terms_to_canonicalize, terms_to_insert, source_map=source_map)

    async def copy_interventions(self):
        """
        Create intervention records
        """
        fields = [
            "lower(prod.product_name) as brand_name",
            "lower(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) as generic_name",
            "ARRAY_AGG(distinct lower(struct.name))::text[] as active_ingredients",
            "ARRAY_REMOVE(ARRAY_AGG(distinct lower(pharma_class.name)), NULL)::text[] as pharmacologic_classes",
        ]
        source_records = PsqlDatabaseClient(SOURCE_DB).select(
            query=ApprovalEtl.get_source_sql(fields)
        )
        source_interventions = [InterventionIntermediate(**r) for r in source_records]

        insert_map = {
            # drugs broken out by combo (more than one active ingredient) or single/compound
            **{
                i.generic_name: {
                    "default_type": BiomedicalEntityType.COMBINATION
                    if len(i.active_ingredients) > 1
                    else BiomedicalEntityType.COMPOUND,
                    "synonyms": compact([i.generic_name, i.brand_name]),
                    **i,
                }
                for i in source_interventions
            },
            # active ingredients for combination drugs
            **{
                ai: {"default_type": BiomedicalEntityType.COMPOUND, "synonyms": [ai]}
                for i in source_interventions
                for ai in i.active_ingredients
                if len(i.active_ingredients) > 1
            },
            # mechanisms / pharmacologic classes
            **{
                pc: {"default_type": BiomedicalEntityType.MECHANISM, "synonyms": [pc]}
                for i in source_interventions
                for pc in i.pharmacologic_classes
            },
        }

        terms_to_insert = list(insert_map.keys())
        terms_to_canonicalize = [
            k for k, v in insert_map.items() if v != BiomedicalEntityType.COMBINATION
        ]

        await BiomedicalEntityEtl(
            "CandidateSelector",
            relation_id_field_map=RelationIdFieldMap(
                comprised_of=RelationConnectInfo(
                    source_field="active_ingredients",
                    dest_field="canonical_id",
                    input_type="connect",
                ),
                parents=RelationConnectInfo(
                    source_field="pharmacologic_classes",
                    dest_field="canonical_id",
                    input_type="connect",
                ),
                synonyms=RelationConnectInfo(
                    source_field="synonyms", dest_field="term", input_type="create"
                ),
            ),
        ).create_records(
            terms_to_canonicalize,
            terms_to_insert,
            insert_map,
        )

    async def copy_documents(self):
        """
        Create regulatory approval records
        """
        fields = [
            "MAX(prod.ndc_product_code) as id",
            "prod.product_name as brand_name",
            "(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) as generic_name",
            "ARRAY_AGG(distinct struct.name)::text[] as active_ingredients",
            "MAX(prod.marketing_status) as application_type",
            "MAX(approval.approval) as approval_date",
            "MAX(approval.type) as agency",
            "array_remove(ARRAY_AGG(distinct metadata.concept_name), NULL) as indications",
            "JSON_AGG(pharma_class.*) as pharmacologic_classes",
            "MAX(label.pdf_url) as url",
        ]
        approvals = PsqlDatabaseClient(SOURCE_DB).select(
            query=ApprovalEtl.get_source_sql(fields)
        )

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
            ]
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
            ]
        )


async def main():
    await ApprovalEtl(document_type="regulatory_approval").copy_all()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.approvals.copy_approvals
            Copies approvals data to postgres
            """
        )
        sys.exit()

    asyncio.run(main())


# update trials set pharmacologic_class=a.pharmacologic_class from regulatory_approvals a where lower(a.generic_name)=lower(trials.intervention);
# update annotations set instance_rollup=a.pharmacologic_class from regulatory_approvals a where lower(term)=lower(a.generic_name);
