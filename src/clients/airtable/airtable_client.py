"""
Client for talking to Airtable
"""
import os
import logging
import polars as pl
from pyairtable import Table

from utils.file import save_as_pickle

API_KEY = os.environ.get("AIRTABLE_API_KEY")


# pylint: disable=C0103
def write_df_to_table(df: pl.DataFrame, base_id: str, table_name: str):
    """
    Write to airtable table

    Args:
        base_id (str): airtable base id
        table_name (str): airtable table name
    """
    records = df.to_dicts()

    if API_KEY is None:
        raise ValueError("Airtable API key not set")

    try:
        # table = create(fields)
        table = Table(API_KEY, base_id, table_name)  # assumed to already exist
        table.batch_create(records, typecast=True)
    except Exception as ex:
        logging.error("Error loading airtable: %s", ex)
        save_as_pickle(records, f"{table_name}.txt")
        raise ex
