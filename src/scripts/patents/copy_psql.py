"""
Utils for copying psql data to BigQuery
"""
import sys
import time
import psycopg2
import logging

from system import initialize

initialize()

from clients.low_level.big_query import (
    DatabaseClient,
    execute_with_retries,
)


def fetch_data_from_postgres(conn, sql_query: str):
    """
    Fetch data from Postgres

    Args:
        conn (psycopg2.connection): Postgres connection
        sql_query (str): SQL query
    """
    with conn.cursor() as cursor:
        cursor.execute(sql_query)
        data = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
    conn.close()
    return columns, data


def copy_from_psql(sql_query: str, new_table_name: str, database: str):
    """
    Copy data from Postgres to BigQuery

    Args:
        sql_query (str): generator SQL query
        new_table_name (str): name of the new table
        database (str): name of the database
    """
    client = DatabaseClient()
    # delete if exists
    client.delete_table(new_table_name)

    conn = psycopg2.connect(
        database=database,
        # user='your_username',
        # password='your_password',
        host="localhost",
        port="5432",
    )
    columns, data = fetch_data_from_postgres(conn, sql_query)

    # recreate
    client.create_table(new_table_name, columns)
    time.sleep(20)  # such a hack

    # add records
    records = [dict(zip(columns, row)) for row in data]
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
