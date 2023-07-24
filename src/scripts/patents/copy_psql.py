"""
Utils for copying psql data to BigQuery
"""
import sys
import psycopg2
import logging

from clients.low_level.big_query import (
    create_bq_table,
    delete_bg_table,
    insert_into_bg_table,
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
    conn = psycopg2.connect(
        database=database,
        # user='your_username',
        # password='your_password',
        host="localhost",
        port="5432",
    )
    columns, data = fetch_data_from_postgres(conn, sql_query)

    # delete if exists
    delete_bg_table(new_table_name)

    # recreate
    create_bq_table(new_table_name, columns)

    # add records
    records = [dict(zip(columns, row)) for row in data]
    insert_into_bg_table(records, new_table_name)


def copy_patent_approvals():
    """
    Copy patent approvals from Postgres to bigquery

    Limitations:
      - Includes only NDA/ANDA, no BLAs nor other country regulatory approvals
    """
    PATENT_FIELDS = [
        "concat('US-', patent_no) as publication_number",
        "app_no as nda_number",
        "p.ndc_product_code as ndc_code",
        "trade_name as brand_name",
        "stem",
        "applicant",
        "p.marketing_status as application_type",
        "approval_date",
        "patent_expire_date",
        "p.description as patent_indication",
        "cd_formula as formula",
        "smiles",
        "cd_molweight as molweight",
        "active_ingredient_count",
        "route",
    ]
    query = f"""
        select {PATENT_FIELDS}
        from ob_patent_view p, product prod, structures s
        where lower(p.trade_name) = lower(prod.product_name)
        AND s.id = p.struct_id
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

    copy_tables: bool = "copy_tables" in sys.argv
    main()
