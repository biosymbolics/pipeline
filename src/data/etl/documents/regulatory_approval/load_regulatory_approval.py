"""
Regulatory approval ETL script
"""

from functools import reduce
import sys
import asyncio
import logging
from typing import Sequence
from pydash import compact
from prisma.enums import BiomedicalEntityType, Source
from prisma.models import (
    Indicatable,
    Intervenable,
    Ownable,
    RegulatoryApproval,
)


from clients.low_level.postgres import PsqlDatabaseClient
from clients.low_level.prisma import prisma_client
from data.etl.types import (
    BiomedicalEntityLoadSpec,
    RelationConnectInfo,
    RelationIdFieldMap,
)
from utils.classes import overrides

from .types import InterventionIntermediate, PharmaClass

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


def get_indication_source_map(
    records: Sequence[dict],
) -> dict[str, dict[str, list[str] | str | bool | BiomedicalEntityType]]:
    return {
        i: {"synonyms": [i], "default_type": BiomedicalEntityType.DISEASE}
        for rec in records
        for i in rec["indications"]
    }


def get_intervention_source_map(
    records: Sequence[dict],
) -> dict[str, dict[str, list[str] | str | bool | BiomedicalEntityType]]:
    """
    Create source map for interventions
    """

    def create_entry(rec: InterventionIntermediate):
        return {
            # main intervention
            rec.name: {
                "active_ingredients": rec.combo_ingredients,
                "pharmacologic_classes": [pc.name for pc in rec.pharmacologic_classes],
                "default_type": rec.default_type,
                "synonyms": compact([rec.generic_name, rec.brand_name]),
            },
            # active ingredients for combination drugs
            **{
                ci: {
                    "default_type": BiomedicalEntityType.COMPOUND,
                    "synonyms": [ci],
                }
                for ci in rec.combo_ingredients
            },
            # mechanisms / pharmacologic classes
            **{
                pc.name: {
                    # set top sorted pharmacologic class as priority
                    "is_priority": i == 0,
                    "default_type": BiomedicalEntityType.MECHANISM,
                    "synonyms": [pc.name],
                }
                for i, pc in enumerate(PharmaClass.sort(rec.pharmacologic_classes))
            },
        }

    return reduce(
        lambda a, b: {**a, **b},
        [create_entry(InterventionIntermediate(**r)) for r in records],
    )


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
        LEFT JOIN pharma_class ON pharma_class.struct_id = struct.id
        LEFT JOIN omop_relationship metadata ON metadata.struct_id = struct.id AND metadata.relationship_name = 'indication'
        WHERE approval.struct_id = struct.id
        AND active_ingredient.struct_id = struct.id
        AND active_ingredient.ndc_product_code = prod.ndc_product_code
        AND p2l.ndc_product_code = prod.ndc_product_code
        AND p2l.label_id = label.id
        AND lower(prod.marketing_status) NOT IN {SUPPRESSION_APPROVAL_TYPES}
        AND approval.approval IS NOT NULL
        AND prod.product_name IS NOT NULL
        GROUP BY prod.product_name
    """

    @overrides(BaseDocumentEtl)
    @staticmethod
    def entity_specs() -> list[BiomedicalEntityLoadSpec]:
        """
        Specs for creating associated biomedical entities (executed by BiomedicalEntityEtl)
        """
        indication_spec = BiomedicalEntityLoadSpec(
            candidate_selector="CompositeCandidateSelector",
            database="drugcentral",
            get_source_map=get_indication_source_map,
            non_canonical_source=Source.FDA,
            sql=RegulatoryApprovalLoader.get_source_sql(
                [
                    "ARRAY_REMOVE(ARRAY_AGG(DISTINCT LOWER(metadata.concept_name)), NULL) AS indications"
                ]
            ),
        )

        intervention_spec = BiomedicalEntityLoadSpec(
            candidate_selector="CompositeCandidateSelector",
            database="drugcentral",
            get_source_map=get_intervention_source_map,
            relation_id_field_map=RelationIdFieldMap(
                comprised_of=RelationConnectInfo(
                    source_field="active_ingredients",
                    dest_field="canonical_id",
                    connect_type="connect",
                ),
                parents=RelationConnectInfo(
                    source_field="pharmacologic_classes",
                    dest_field="canonical_id",
                    connect_type="connect",
                ),
                synonyms=RelationConnectInfo(
                    # todo: connectOrCreate when supported
                    # https://github.com/RobertCraigie/prisma-client-py/issues/754
                    source_field="synonyms",
                    dest_field="term",
                    connect_type="create",
                ),
            ),
            get_terms_to_canonicalize=lambda sm: (
                [
                    k
                    for k, v in sm.items()
                    if v["default_type"] != BiomedicalEntityType.COMBINATION
                ],
                None,  # no vectors
            ),
            non_canonical_source=Source.FDA,
            sql=RegulatoryApprovalLoader.get_source_sql(
                [
                    "LOWER(prod.product_name) AS brand_name",
                    "LOWER(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) AS generic_name",
                    "ARRAY_AGG(DISTINCT LOWER(struct.name))::text[] AS active_ingredients",
                    "JSON_AGG(DISTINCT pharma_class.*) AS pharmacologic_classes",
                ]
            ),
        )
        return [intervention_spec, indication_spec]

    @overrides(BaseDocumentEtl)
    async def delete_documents(self):
        client = await prisma_client(600)
        await Intervenable.prisma(client).query_raw(
            "DELETE FROM intervenable WHERE regulatory_approval_id IS NOT NULL"
        )
        await Indicatable.prisma(client).query_raw(
            "DELETE FROM indicatable WHERE regulatory_approval_id IS NOT NULL"
        )
        await Ownable.prisma(client).query_raw(
            "DELETE FROM ownable WHERE regulatory_approval_id IS NOT NULL"
        )
        await RegulatoryApproval.prisma(client).delete_many()

    @overrides(BaseDocumentEtl)
    async def copy_documents(self):
        """
        Create regulatory approval records
        """
        fields = [
            "MAX(prod.ndc_product_code) AS id",
            "prod.product_name AS brand_name",
            "LOWER(MAX(approval.applicant)) AS applicant",
            "(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) AS generic_name",
            "ARRAY_AGG(distinct struct.name)::text[] AS active_ingredients",
            "MAX(prod.marketing_status) AS application_type",
            "MAX(approval.approval)::TIMESTAMP AS approval_date",
            "MAX(approval.type) AS agency",
            "ARRAY_REMOVE(ARRAY_AGG(distinct metadata.concept_name), NULL) AS indications",
            "MAX(label.pdf_url) AS url",
        ]
        approvals = await PsqlDatabaseClient(SOURCE_DB).select(
            query=RegulatoryApprovalLoader.get_source_sql(fields)
        )

        client = await prisma_client(600)

        logger.info("Creating %s regulatory approval records", len(approvals))
        await RegulatoryApproval.prisma(client).create_many(
            data=[
                {
                    "id": a["id"],
                    "agency": a["agency"],
                    "approval_date": a["approval_date"],
                    "application_type": a["application_type"],
                    "url": a["url"],
                }
                for a in approvals
            ],
            skip_duplicates=True,
        )

        # create owner records (aka applicants)
        await Ownable.prisma(client).create_many(
            data=[
                {
                    "name": (a["applicant"] or "unknown").lower(),
                    "canonical_name": (a["applicant"] or "unknown").lower(),
                    "instance_rollup": (a["applicant"] or "unknown").lower(),
                    "category_rollup": (a["applicant"] or "unknown").lower(),
                    "regulatory_approval_id": a["id"],
                }
                for a in approvals
            ],
            skip_duplicates=True,
        )

        logger.info("Creating regulatory approval indicatable records")
        await Indicatable.prisma().create_many(
            data=[
                {
                    "name": i.lower(),
                    "canonical_name": i.lower(),
                    "regulatory_approval_id": a["id"],
                }
                for a in approvals
                for i in a["indications"]
            ]
        )

        logger.info("Creating regulatory approval intervenable records")
        await Intervenable.prisma().create_many(
            data=[
                {
                    "name": (a["generic_name"] or a["brand_name"]).lower(),
                    "canonical_name": a["generic_name"].lower(),
                    "is_primary": True,
                    "regulatory_approval_id": a["id"],
                }
                for a in approvals
            ]
        )


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.documents.regulatory_approval.load_regulatory_approval [--update]
            Copies approvals data to postgres
            """
        )
        sys.exit()

    is_update = "--update" in sys.argv

    asyncio.run(
        RegulatoryApprovalLoader(document_type="regulatory_approval").copy_all(
            is_update
        )
    )
