"""
Utils for copying approvals data
"""
import sys

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient


def copy_patent_approvals():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)

    Limitations:
      - Includes only NDA/ANDA, no BLAs nor other country regulatory approvals
    """
    PATENT_FIELDS = [
        "concat('US-', patent_no) as publication_number",
        "(array_agg(appl_no))[1] as nda_number",
        "(array_agg(prod.ndc_product_code))[1] as ndc_code",
        "(array_agg(trade_name))[1] as brand_name",
        "(ARRAY_TO_STRING(array_agg(distinct s.name), '+')) as generic_name",
        "(array_agg(stem))[1] as stem",
        "(array_agg(applicant))[1] as applicant",
        "(array_agg(prod.marketing_status))[1] as application_type",
        "(array_agg(approval_date))[1] as approval_date",
        "(array_agg(patent_expire_date))[1] as patent_expire_date",
        "(array_agg(pv.description))[1] as patent_indication",
        "(array_agg(cd_formula))[1] as formula",
        "(array_agg(smiles))[1] as smiles",
        "(array_agg(cd_molweight))[1] as molweight",
        "(array_agg(active_ingredient_count))[1] as active_ingredient_count",
        "(array_agg(pv.route))[1] as route",
    ]
    source_sql = f"""
        select {", ".join(PATENT_FIELDS)}
        from ob_patent_view pv, product prod, structures s
        where lower(pv.trade_name) = lower(prod.product_name)
        AND s.id = pv.struct_id
        group by pv.patent_no
    """
    source_db = "drugcentral"
    dest_db = "patents"
    dest_table_name = "patent_approvals"
    PsqlDatabaseClient.copy_between_db(
        source_db=source_db,
        source_sql=source_sql,
        dest_db=dest_db,
        dest_table_name=dest_table_name,
        truncate_if_exists=True,
    )


def main():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)
    """
    copy_patent_approvals()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 copy_psql.py\nCopies approvals data to postgres")
        sys.exit()

    main()
