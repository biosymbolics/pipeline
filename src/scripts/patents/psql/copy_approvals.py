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


def copy_patent_to_regulatory_approval():
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


def main():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)
    """
    copy_all_approvals()
    copy_patent_to_regulatory_approval()


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
