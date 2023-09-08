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


def copy_all_approvals():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)
    """
    PATENT_FIELDS = [
        "max(prod_approval.appl_no) as regulatory_application_number",
        "max(prod.ndc_product_code) as ndc_code",
        "max(prod.product_name) as brand_name",
        "(ARRAY_TO_STRING(array_agg(distinct struct.name), '+')) as generic_name",  # or product.generic_name
        "ARRAY_AGG(distinct struct.name) as active_ingredients",
        "max(struct.stem) as stem",
        "max(prod_approval.applicant) as applicant",
        "max(prod.marketing_status) as application_type",
        "max(prod_approval.approval_date) as approval_date",
        "max(label_section.text) as patent_indication",
        "max(distinct struct.cd_formula) as formula",
        "max(distinct struct.smiles) as smiles",
        "max(distinct struct.lipinski) as lipinski",
        "max(prod_approval.route) as route",
        "max(label.pdf_url) as label_url",
        "'' as normalized_applicant",
    ]
    # faers? ddi?
    source_sql = f"""
        select {", ".join(PATENT_FIELDS)}
        from
        structures struct,
        struct2obprod s2p,
        prd2label p2l,
        label,
        section label_section,
        ob_product prod_approval,
        product prod
        where prod_approval.id = s2p.prod_id
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
    )
    client = PsqlDatabaseClient()
    client.execute_query(
        """
        update {REGULATORY_APPROVAL_TABLE} set normalized_applicant=sm.term from
        synonym_map sm where sm.synonym = lower(applicant)
        """
    )
    client.create_indices(
        [
            {
                "table": REGULATORY_APPROVAL_TABLE,
                "column": "regulatory_application_number",
            },
            {
                "table": REGULATORY_APPROVAL_TABLE,
                "column": "active_ingredients",
                "is_gin": True,
            },
            {
                "table": REGULATORY_APPROVAL_TABLE,
                "column": "normalized_applicant",
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
        prod_approval.appl_no as regulatory_application_number
        from
        product prod,
        ob_product prod_approval,
        ob_patent_view patents
        where prod_approval.trade_name = prod.product_name
        AND lower(patents.trade_name) = lower(prod.product_name)
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
    query = """
        select *
        from {REGULATORY_APPROVAL_TABLE} approvals,
        aggregated_annotations as a,
        applications as patent_applications
        LEFT JOIN synonym_map as sm ON sm.synonym = approvals.generic_name
        where aggregated_annotations.publication_number = patent_applications.publication_number
        AND approvals.active_ingredients @> a.terms -- TODO this could FP
        AND t.normalized_appliant = any(a.terms)
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
    copy_patent_to_approval()


def main():
    copy_approvals()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.patents.psql.copy_approvals
            Copies approvals data to postgres
        """
        )
        sys.exit()

    main()
