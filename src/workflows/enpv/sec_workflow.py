"""
SEC pipeline for eNPV estimation
"""
from datetime import datetime
import logging

from common.clients.airtable.airtable_client import write_df_to_table
from sources.sec.product_pipeline import get_pipeline_by_ticker

DEFAULT_BASE_ID = "appcXwgAM75mx9sGi"


def run_sec_pipeline(ticker: str):
    """
    Run SEC pipeline
    """
    start_date = datetime(2018, 1, 1)
    try:
        pipeline = get_pipeline_by_ticker(ticker, start_date)
        write_df_to_table(pipeline, base_id=DEFAULT_BASE_ID, table_name=ticker.lower())
    except Exception as ex:
        logging.error("Error running pipeline: %s", ex)


def main():
    """
    Main
    """
    run_sec_pipeline("PFE")


if __name__ == "__main__":
    main()
