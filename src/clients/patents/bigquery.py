from google.cloud import bigquery
from psycopg2 import sql, connect
from datetime import datetime
import json
import logging

BIOMEDICAL_IPC_CODES = ["A61", "C07", "C12", "G01N"]
IPC_RE = r"^({})".format("|".join(BIOMEDICAL_IPC_CODES))


def __bigquery_to_file(query: str, filename: str):
    """
    Query bigquery

    Args:
        query (str): SQL query
    """
    # Create a client
    client = bigquery.Client()

    logging.info("Starting query: %s", query)
    query_job = client.query(query)

    # Get a row iterator to stream the results
    logging.info("Starting to stream results")
    rows = query_job.result(page_size=500)  # Fetch 500 rows at a time

    # Write results to a file
    logging.info("Starting to write file %s", filename)
    date = datetime.now().strftime("%Y-%m-%d")
    with open(f"{filename}-{date}.json", "w") as f:
        for row in rows:
            f.write(json.dumps(dict(row)) + "\n")

    logging.info("Finished writing file %s", filename)


def persist_to_sql(file_location: str):
    """
    Read the query results from a file and create sql insert statements for persistence into a PSQL database.
    """
    # Connect to your postgres DB
    conn = connect("dbname=biosymbolics user=postgres password=secret")

    # Open a cursor to perform database operations
    cur = conn.cursor()

    # Read the file and insert rows to SQL
    with open(file_location, "r") as f:
        for line in f:
            row_dict = json.loads(line.strip())

            # Prepare SQL INSERT statement
            insert_sql = sql.SQL(
                "INSERT INTO publications ({}) VALUES ({})".format(
                    sql.SQL(",").join(map(sql.Identifier, row_dict.keys())),
                    sql.SQL(",").join(map(sql.Placeholder, row_dict.keys())),
                )
            )

            # Execute the SQL statement
            cur.execute(insert_sql, row_dict)

    # Commit the changes and close the connection
    conn.commit()
    cur.close()
    conn.close()


def copy_gpr_publications():
    """
    Query GPR publications for biomedical patents

    Store results in a file
    Then create sql insert statements for persistence into a PSQL database
    """
    query = (
        "SELECT * FROM `patents-public-data.google_patents_research.publications`, "
        "UNNEST(cpc) as cpc "
        f"WHERE REGEXP_CONTAINS(cpc.code, {IPC_RE})"
    )

    __bigquery_to_file(query, "gpr-publications")
