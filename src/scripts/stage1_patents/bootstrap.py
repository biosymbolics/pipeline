"""
Functions to initialize the patents database
"""

import asyncio
import logging
import sys
from data.etl.documents.trial.vectorize_trials import TrialVectorizer

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import (
    PATENT_VECTOR_TABLE,
    REGULATORY_APPROVAL_VECTOR_TABLE,
    TRIAL_VECTOR_TABLE,
)
from data.etl.documents.patent.vectorize_patents import PatentVectorizer

from .prep_bq_patents import copy_patent_tables
from .import_bq_patents import copy_bq_to_psql


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


async def create_funcs():
    re_escape_sql = r"""
        CREATE OR REPLACE FUNCTION escape_regex_chars(text)
        RETURNS text
        LANGUAGE sql IMMUTABLE STRICT PARALLEL SAFE AS
        $func$
        SELECT regexp_replace($1, '([!$()*+.:<=>?[\\\]^{|}-])', '\\\1', 'g')
        $func$;

        CREATE OR REPLACE FUNCTION zip(anyarray, anyarray)
        RETURNS SETOF anyarray LANGUAGE SQL AS
        $func$
        SELECT ARRAY[a,b] FROM (SELECT unnest($1) AS a, unnest($2) AS b) x;
        $func$;
    """

    await PsqlDatabaseClient().execute_query(re_escape_sql)
    return


async def vectorize_documents():
    """
    Vectorize patents and store in postgres
    """
    for db, table in [
        ("patents", PATENT_VECTOR_TABLE),
        ("drugcentral", REGULATORY_APPROVAL_VECTOR_TABLE),
        ("aact", TRIAL_VECTOR_TABLE),
    ]:
        await PsqlDatabaseClient(db).create_table(
            table, {"id": "text", "vector": "vector(768)"}, exists_ok=True
        )

    # create vectors for patents. this will take 4+ hours (depends upon GPU & memory)
    await PatentVectorizer()()
    await TrialVectorizer()()


async def main():
    """
    Copy tables from patents-public-data to a local dataset.
    Order matters.
    """
    await create_funcs()
    await copy_patent_tables()
    await copy_bq_to_psql()
    await vectorize_documents()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
                Usage: python3 -m scripts.stage1_patents.bootstrap
                Creates initial patent data, which is later ETL'd into biosym
                Run rarely. Will take a long time (esp patent vectorization)
            """
        )
        sys.exit()

    asyncio.run(main())
