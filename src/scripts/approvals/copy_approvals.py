"""
Utils for copying approvals data
"""
import re
import sys

from pydash import compact, uniq

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from constants.patterns.intervention import PRIMARY_MECHANISM_BASE_TERMS
from constants.core import REGULATORY_APPROVAL_TABLE
from utils.re import get_or_re


SOURCE_DB = "drugcentral"
DEST_DB = "patents"
SEARCH_FIELDS = {
    # "applicant": "coalesce(applicant, '')",
    "active_ingredients": "ARRAY_TO_STRING(active_ingredients, '|| " " ||')",
    "brand_name": "coalesce(brand_name, '')",
    "generic_name": "coalesce(generic_name, '')",
    "indications": "ARRAY_TO_STRING(indications, '|| " " ||')",
    "pharmacologic_classes": "ARRAY_TO_STRING(pharmacologic_classes, '|| " " ||')",
}


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

    if len(pharmacologic_classes) == 0:
        return None

    prioritized = sorted(pharmacologic_classes, key=get_priority, reverse=True)

    print("HIHI", prioritized)
    return prioritized[0]["name"].lower()


def copy_all_approvals():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)
    """
    PATENT_FIELDS = [
        "MAX(prod.ndc_product_code) as ndc_code",
        "prod.product_name as brand_name",
        "(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), ' / ')) as generic_name",  # or product.generic_name
        "ARRAY_AGG(distinct struct.name)::text[] as active_ingredients",
        # "MAX(prod_approval.applicant) as applicant",
        "MAX(prod.marketing_status) as application_type",
        "MAX(approval.approval) as approval_date",
        "MAX(approval.type) as regulatory_agency",
        "array_remove(ARRAY_AGG(distinct metadata.concept_name), NULL) as indications",
        "JSON_AGG(pharma_class.*) as pharmacologic_classes",
        "MAX(label.pdf_url) as label_url",
    ]
    source_sql = f"""
        select {", ".join(PATENT_FIELDS)}
        from
        approval,
        active_ingredient,
        product prod,
        prd2label p2l,
        label,
        section label_section,
        structures struct
        LEFT JOIN pharma_class on pharma_class.struct_id = struct.id
        LEFT JOIN omop_relationship metadata on metadata.struct_id = struct.id and metadata.relationship_name = 'indication'
        where approval.struct_id = struct.id -- TODO: combo drugs??
        AND active_ingredient.struct_id = struct.id
        AND active_ingredient.ndc_product_code = prod.ndc_product_code
        AND p2l.ndc_product_code = prod.ndc_product_code
        AND p2l.label_id = label.id
        AND label_section.label_id = p2l.label_id
        AND label_section.title = 'INDICATIONS & USAGE SECTION'
        group by prod.product_name
    """
    PsqlDatabaseClient.copy_between_db(
        source_db=SOURCE_DB,
        source_sql=source_sql,
        dest_db=DEST_DB,
        dest_table_name=REGULATORY_APPROVAL_TABLE,
        truncate_if_exists=True,
        transform_schema=lambda schema: {
            **schema,
            "pharmacologic_classes": "text[]",
            "pharmacologic_class": "text",
            "text_search": "tsvector",
        },
        transform=lambda batch, _: [
            {
                **row,
                "pharmacologic_class": get_preferred_pharmacologic_class(
                    compact(uniq(row["pharmacologic_classes"]))
                ),
                "pharmacologic_classes": uniq(
                    [
                        pc["name"].lower()
                        for pc in row["pharmacologic_classes"]
                        if pc is not None
                    ]
                ),
            }
            for row in batch
        ],
    )
    client = PsqlDatabaseClient()
    vector_sql = ("|| ' ' ||").join(SEARCH_FIELDS.values())
    client.execute_query(
        f"""
        -- update {REGULATORY_APPROVAL_TABLE} set normalized_applicant=sm.term from
        -- synonym_map sm where sm.synonym = lower(applicant);

        UPDATE {REGULATORY_APPROVAL_TABLE} SET text_search = to_tsvector('english', {vector_sql});
        """
    )
    client.create_indices(
        [
            {
                "table": REGULATORY_APPROVAL_TABLE,
                "column": "ndc_code",
            },
            {
                "table": REGULATORY_APPROVAL_TABLE,
                "column": "text_search",
                "is_gin": True,
            },
        ]
    )


def copy_approvals():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)
    """
    copy_all_approvals()


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

    copy_approvals()
