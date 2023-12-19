"""
Utils for copying approvals data
"""
import sys

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import (
    REGULATORY_APPROVAL_TABLE,
    PATENT_TO_REGULATORY_APPROVAL_TABLE,
)

SOURCE_DB = "drugcentral"
DEST_DB = "patents"
SEARCH_FIELDS = {
    "applicant": "coalesce(applicant, '')",
    "active_ingredients": "ARRAY_TO_STRING(active_ingredients, '|| " " ||')",
    "brand_name": "coalesce(brand_name, '')",
    "generic_name": "coalesce(generic_name, '')",
    "indications": "ARRAY_TO_STRING(indications, '|| " " ||')",
}


def copy_all_approvals():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)

    NOTE: drugcentral is a bit if a sh*tshow. there is lots of partly duplicated data
    and it doesn't really add up, e.g. why are there 1136 distinct applicants in `approval`
    and 1772 in `ob_product`?
    """
    PATENT_FIELDS = [
        "max(prod_approval.appl_no) as application_number",
        "max(prod.ndc_product_code) as ndc_code",
        "max(prod.product_name) as brand_name",
        "(ARRAY_TO_STRING(ARRAY_AGG(distinct struct.name), '+')) as generic_name",  # or product.generic_name
        "ARRAY_AGG(distinct struct.name)::text[] as active_ingredients",
        "max(prod_approval.applicant) as applicant",
        "max(prod.marketing_status) as application_type",
        "ARRAY_AGG(distinct prod.marketing_status) as application_types",
        "ARRAY_AGG(distinct approval.approval) as approval_dates",
        "ARRAY_AGG(distinct approval.type) as regulatory_agencies",
        "ARRAY_AGG(distinct label_section.text) as indications",
        "ARRAY_AGG(distinct prod_approval.route) as routes",
        "max(label.pdf_url) as label_url",
        "'' as normalized_applicant",
    ]
    source_sql = f"""
        select {", ".join(PATENT_FIELDS)}
        from
        structures struct,
        struct2obprod s2p,
        prd2label p2l,
        label,
        section label_section,
        ob_product prod_approval,
        product prod,
        approval
        where prod_approval.id = s2p.prod_id
        AND approval.struct_id = struct.id -- TODO: combo drugs??
        AND prod.product_name = prod_approval.trade_name
        AND s2p.struct_id = struct.id
        AND p2l.ndc_product_code = prod.ndc_product_code
        AND label.id = p2l.label_id
        AND label_section.label_id = label.id
        AND label_section.title = 'INDICATIONS & USAGE SECTION'
        group by prod_approval.trade_name;
    """
    PsqlDatabaseClient.copy_between_db(
        source_db=SOURCE_DB,
        source_sql=source_sql,
        dest_db=DEST_DB,
        dest_table_name=REGULATORY_APPROVAL_TABLE,
        truncate_if_exists=True,
        transform_schema=lambda schema: {**schema, "text_search": "tsvector"},
    )
    client = PsqlDatabaseClient()
    vector_sql = ("|| ' ' ||").join(SEARCH_FIELDS.values())
    client.execute_query(
        f"""
        update {REGULATORY_APPROVAL_TABLE} set normalized_applicant=sm.term from
        synonym_map sm where sm.synonym = lower(applicant);

        UPDATE {REGULATORY_APPROVAL_TABLE} SET text_search = to_tsvector('english', {vector_sql});
        """
    )
    client.create_indices(
        [
            {
                "table": REGULATORY_APPROVAL_TABLE,
                "column": "application_number",
            },
            {
                "table": REGULATORY_APPROVAL_TABLE,
                "column": "text_search",
                "is_gin": True,
            },
        ]
    )


def copy_direct_patent_to_approval():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)
    """
    source_sql = f"""
        select
        concat('US-', patent_no) as publication_number,
        prod_approval.appl_no as application_number
        from
        product prod,
        ob_product prod_approval,
        ob_patent_view patents
        where prod_approval.trade_name = prod.product_name
        AND lower(patents.trade_name) = lower(prod.product_name)
        group by concat('US-', patent_no), prod_approval.appl_no
    """
    PsqlDatabaseClient.copy_between_db(
        source_db=SOURCE_DB,
        source_sql=source_sql,
        dest_db=DEST_DB,
        dest_table_name=PATENT_TO_REGULATORY_APPROVAL_TABLE,
        truncate_if_exists=True,
    )


def copy_indirect_patent_to_approval():
    """
    Map patents to regulatory approvals based on
    1) intervention/ingredient name match
    2) sponsor match
    """
    client = PsqlDatabaseClient()
    query = f"""
        select
        patent_app.publication_number as publication_number,
        approvals.application_number as regulatory_application_number
        from
        applications as patent_app,
        aggregated_annotations as a,
        {REGULATORY_APPROVAL_TABLE} approvals
        where a.publication_number = patent_app.publication_number
        AND {REGULATORY_APPROVAL_TABLE}.active_ingredients @> a.terms -- TODO this could FP
        AND coalesce(nullif({REGULATORY_APPROVAL_TABLE}.normalized_applicant, ''), lower({REGULATORY_APPROVAL_TABLE}.applicant)) = any(a.terms)
    """
    client.select_insert_into_table(query, PATENT_TO_REGULATORY_APPROVAL_TABLE)


def copy_patent_to_approval():
    """
    Copy map between patents and regulatory approvals
    """
    copy_direct_patent_to_approval()
    copy_indirect_patent_to_approval()

    client = PsqlDatabaseClient()
    client.create_indices(
        [
            {
                "table": PATENT_TO_REGULATORY_APPROVAL_TABLE,
                "column": "publication_number",
            },
            {
                "table": PATENT_TO_REGULATORY_APPROVAL_TABLE,
                "column": "regulatory_application_number",
            },
        ]
    )


def copy_approvals():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)
    """
    copy_all_approvals()
    # copy_patent_to_approval()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.approvals.copy_approvals
            Copies approvals data to postgres
        """
        )
        sys.exit()

    copy_approvals()
