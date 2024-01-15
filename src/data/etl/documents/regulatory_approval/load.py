"""
Regulatory approval ETL script
"""
from datetime import datetime
import re
import sys
import asyncio
import logging
from typing import Sequence
from pydash import compact
from prisma.enums import BiomedicalEntityType, Source
from prisma.models import (
    Indicatable,
    Intervenable,
    RegulatoryApproval,
)


from clients.low_level.postgres import PsqlDatabaseClient
from constants.patterns.intervention import PRIMARY_MECHANISM_BASE_TERMS
from data.etl.types import (
    BiomedicalEntityLoadSpec,
    RelationConnectInfo,
    RelationIdFieldMap,
)
from utils.re import get_or_re

from .types import InterventionIntermediate

from ..base_document import BaseDocumentEtl


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


def get_indication_source_map(records: Sequence[dict]) -> dict:
    return {
        i: {"synonyms": [i], "default_type": BiomedicalEntityType.DISEASE}
        for rec in records
        for i in rec["indications"]
    }


def get_intervention_source_map(records: Sequence[dict]) -> dict:
    i_recs = [InterventionIntermediate(**r) for r in records]
    return {
        rec["intervention"]: {
            # drugs broken out by combo (more than one active ingredient) or single/compound
            **{
                rec.generic_name: {
                    "default_type": BiomedicalEntityType.COMBINATION
                    if len(rec.active_ingredients) > 1
                    else BiomedicalEntityType.COMPOUND,
                    "synonyms": compact([rec.generic_name, rec.brand_name]),
                    **rec,
                }
                for rec in i_recs
            },
            # active ingredients for combination drugs
            **{
                ai: {
                    "default_type": BiomedicalEntityType.COMPOUND,
                    "synonyms": [ai],
                }
                for rec in i_recs
                for ai in rec.active_ingredients
                if len(rec.active_ingredients) > 1
            },
            # mechanisms / pharmacologic classes
            **{
                pc: {
                    "default_type": BiomedicalEntityType.MECHANISM,
                    "synonyms": [pc],
                }
                for rec in i_recs
                for pc in rec.pharmacologic_classes
            },
        }
        for rec in i_recs
    }


class RegulatoryApprovalLoader(BaseDocumentEtl):
    """
    Load regulatory approvals and associated entities
    """

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

    @staticmethod
    def entity_specs() -> list[BiomedicalEntityLoadSpec]:
        """
        Specs for creating associated biomedical entities (executed by BiomedicalEntityEtl)
        """
        indication_spec = BiomedicalEntityLoadSpec(
            candidate_selector="CandidateSelector",
            get_source_map=get_indication_source_map,
            non_canonical_source=Source.FDA,
            sql=RegulatoryApprovalLoader.get_source_sql(
                [
                    "array_remove(ARRAY_AGG(distinct lower(metadata.concept_name)), NULL) as indications"
                ]
            ),
        )

        intervention_spec = BiomedicalEntityLoadSpec(
            candidate_selector="CandidateSelector",
            get_source_map=get_intervention_source_map,
            relation_id_field_map=RelationIdFieldMap(
                comprised_of=RelationConnectInfo(
                    source_field="active_ingredients",
                    dest_field="canonical_id",
                    input_type="set",
                ),
                parents=RelationConnectInfo(
                    source_field="pharmacologic_classes",
                    dest_field="canonical_id",
                    input_type="set",
                ),
                synonyms=RelationConnectInfo(
                    source_field="synonyms", dest_field="term", input_type="create"
                ),
            ),
            get_terms_to_canonicalize=lambda sm: [
                k for k, v in sm.items() if v != BiomedicalEntityType.COMBINATION
            ],
            non_canonical_source=Source.FDA,
            sql=RegulatoryApprovalLoader.get_source_sql(
                [
                    "lower(prod.product_name) as brand_name",
                    "lower(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) as generic_name",
                    "ARRAY_AGG(distinct lower(struct.name))::text[] as active_ingredients",
                    "ARRAY_REMOVE(ARRAY_AGG(distinct lower(pharma_class.name)), NULL)::text[] as pharmacologic_classes",
                ]
            ),
        )
        return [indication_spec, intervention_spec]

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
            "ARRAY_REMOVE(ARRAY_AGG(distinct metadata.concept_name), NULL) as indications",
            "JSON_AGG(pharma_class.*) as pharmacologic_classes",
            "MAX(label.pdf_url) as url",
        ]
        approvals = await PsqlDatabaseClient(SOURCE_DB).select(
            query=RegulatoryApprovalLoader.get_source_sql(fields)
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
                }
                for a in approvals
            ],
            skip_duplicates=True,
        )

        # create "indicatable" records, those that map approval to a canonical indication
        await Indicatable.prisma().create_many(
            data=[
                {
                    "name": i,
                    "canonical_name": i,
                    "instance_rollup": i,
                    "regulatory_approval_id": a["id"],
                }
                for a in approvals
                for i in a["indications"]
            ]
        )

        # create "intervenable" records, those that map approval to a canonical intervention
        await Intervenable.prisma().create_many(
            data=[
                {
                    "name": a["generic_name"] or a["brand_name"],
                    "canonical_name": get_preferred_pharmacologic_class(
                        a["pharmacologic_classes"]
                    )
                    or "",
                    "instance_rollup": get_preferred_pharmacologic_class(
                        a["pharmacologic_classes"]
                    ),  # TODO: should be done via biomedical_entity/UMLS mapping step
                    "is_primary": True,
                    "regulatory_approval_id": a["id"],
                }
                for a in approvals
            ]
        )


async def main():
    await RegulatoryApprovalLoader(document_type="regulatory_approval").copy_all()


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
