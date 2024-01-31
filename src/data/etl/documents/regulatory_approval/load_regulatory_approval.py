"""
Regulatory approval ETL script
"""
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


def get_indication_source_map(records: Sequence[dict]) -> dict:
    return {
        i: {"synonyms": [i], "default_type": BiomedicalEntityType.DISEASE}
        for rec in records
        for i in rec["indications"]
    }


def get_intervention_source_map(records: Sequence[dict]) -> dict[str, dict]:
    i_recs = [InterventionIntermediate(**r) for r in records]

    def create_sm_entry(rec: InterventionIntermediate):
        is_combo = len(rec.active_ingredients) > 1
        combo_ingredients = rec.active_ingredients if is_combo else []
        main_default_type = (
            BiomedicalEntityType.COMBINATION
            if is_combo
            else BiomedicalEntityType.COMPOUND
        )

        return {
            "active_ingredients": combo_ingredients,
            "pharmacologic_classes": [pc.name for pc in rec.pharmacologic_classes],
            # drugs broken out by combo (more than one active ingredient) or single/compound
            **{
                rec.generic_name: {
                    "default_type": main_default_type,
                    "synonyms": compact([rec.generic_name, rec.brand_name]),
                    **rec.__dict__,
                }
            },
            # active ingredients for combination drugs
            **{
                ci: {
                    "default_type": BiomedicalEntityType.COMPOUND,
                    "synonyms": [ci],
                }
                for ci in combo_ingredients
            },
            # mechanisms / pharmacologic classes
            **{
                pc.name: {
                    # set top sorted pharmacologic class as priority
                    "is_priority": i == 0,  # todo - not actually used
                    "default_type": BiomedicalEntityType.MECHANISM,
                    "synonyms": [pc.name],
                }
                for i, pc in enumerate(PharmaClass.sort(rec.pharmacologic_classes))
            },
        }

    return {rec.intervention: create_sm_entry(rec) for rec in i_recs}


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

    @overrides(BaseDocumentEtl)
    @staticmethod
    def entity_specs() -> list[BiomedicalEntityLoadSpec]:
        """
        Specs for creating associated biomedical entities (executed by BiomedicalEntityEtl)
        """
        indication_spec = BiomedicalEntityLoadSpec(
            candidate_selector="CandidateSelector",
            database="drugcentral",
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
            database="drugcentral",
            get_source_map=get_intervention_source_map,
            relation_id_field_map=RelationIdFieldMap(
                comprised_of=RelationConnectInfo(
                    # "active ingredients" not in source?
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
                    source_field="synonyms",
                    dest_field="term",
                    connect_type="create",
                ),
            ),
            get_terms_to_canonicalize=lambda sm: (
                [
                    k
                    for sub in sm.values()
                    if isinstance(sub, dict)  # is dict, therefore a parent spec
                    for k, v in sub.items()
                    if v != BiomedicalEntityType.COMBINATION
                ],
                None,
            ),
            non_canonical_source=Source.FDA,
            sql=RegulatoryApprovalLoader.get_source_sql(
                [
                    "lower(prod.product_name) as brand_name",
                    "lower(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) as generic_name",
                    "ARRAY_AGG(distinct lower(struct.name))::text[] as active_ingredients",
                    "JSON_AGG(distinct pharma_class.*) as pharmacologic_classes",
                ]
            ),
        )
        return [intervention_spec, indication_spec]

    @overrides(BaseDocumentEtl)
    async def delete_documents(self):
        client = await prisma_client(600)
        await RegulatoryApproval.prisma(client).delete_many()

    @overrides(BaseDocumentEtl)
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
            "MAX(approval.approval)::TIMESTAMP as approval_date",
            "MAX(approval.type) as agency",
            "ARRAY_REMOVE(ARRAY_AGG(distinct metadata.concept_name), NULL) as indications",
            "MAX(label.pdf_url) as url",
        ]
        approvals = await PsqlDatabaseClient(SOURCE_DB).select(
            query=RegulatoryApprovalLoader.get_source_sql(fields)
        )

        client = await prisma_client(600)

        # create approval records
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

        # create "indicatable" records, those that map approval to a canonical indication
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

        # create "intervenable" records, those that map approval to a canonical intervention
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
            Usage: python3 -m data.etl.documents.regulatory_approval.load [--update]
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
