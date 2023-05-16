"""
Client for talking to Airtable
"""
import os
import logging
import polars as pl
from pyairtable import Table

API_KEY = os.environ["AIRTABLE_API_KEY"]

# def create_table(base_id: str, table_name: str, schema):


def write_df_to_table(df: pl.DataFrame, base_id: str, table_name: str):
    """
    Write to airtable table
    :param str base_id: airtable base id
    :param str table_name: airtable table name
    """
    records = df.to_dicts()

    try:
        # table = create(fields)
        table = Table(API_KEY, base_id, table_name)  # assumed to already exist
        table.batch_create(records, typecast=True)
    except Exception as ex:
        logging.error("Error loading airtable: %s", ex)
