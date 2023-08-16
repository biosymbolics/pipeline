"""
Utils for copying approvals data
"""
import sys

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from clients.low_level.database import execute_with_retries


def copy_from_psql(sql_query: str, new_table_name: str, database: str):
    """
    Copy data from Postgres to BigQuery

    Args:
        sql_query (str): generator SQL query
        new_table_name (str): name of the new table
        database (str): name of the database
    """
    client = PsqlDatabaseClient()
    # truncate if exists
    client.truncate_table(new_table_name)

    # pull records from other db
    results = PsqlDatabaseClient(database).execute_query(sql_query)
    records = [dict(row) for row in results["data"]]

    # recreate
    client.create_table(
        new_table_name, results["columns"], exists_ok=True, truncate_if_exists=True
    )

    # add records
    client.insert_into_table(records, new_table_name)
    execute_with_retries(lambda: client.insert_into_table(records, new_table_name))


def copy_patent_approvals():
    """
    Copy patent approvals from Postgres to bigquery

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
    query = f"""
        select {", ".join(PATENT_FIELDS)}
        from ob_patent_view pv, product prod, structures s
        where lower(pv.trade_name) = lower(prod.product_name)
        AND s.id = pv.struct_id
        group by pv.patent_no
    """
    database = "drugcentral"
    new_table_name = "patent_approvals"
    copy_from_psql(query, new_table_name, database)


def main():
    """
    Copy data from Postgres to BigQuery
    """
    copy_patent_approvals()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 copy_psql.py\nCopies psql data to BigQuery")
        sys.exit()

    main()
