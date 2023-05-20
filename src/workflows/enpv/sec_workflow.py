"""
SEC pipeline for eNPV estimation
"""
from datetime import datetime
import logging

from common.clients.airtable.airtable_client import write_df_to_table
from common.clients.llama_index.visualization import visualize_network
from sources.sec.rd_pipeline import get_pipeline_by_ticker

DEFAULT_BASE_ID = "appcXwgAM75mx9sGi"


def run_sec_pipeline(ticker: str):
    """
    Run SEC pipeline
    """
    start_date = datetime(2022, 1, 1)
    try:
        pipeline = get_pipeline_by_ticker(ticker, start_date, datetime.now(), "SEARCH")
        write_df_to_table(pipeline, base_id=DEFAULT_BASE_ID, table_name=ticker.lower())
    except Exception as ex:
        logging.error("Error running pipeline: %s", ex)
        raise ex


def main():
    """
    Main
    """
    # PFE, JNJ, NVS (Novartis), RHHBY (Roche), APPV, MRK, Bristol Myers Squibb (BMY)
    run_sec_pipeline("BMY")
    visualize_network("BMY", "2021-12-31")


if __name__ == "__main__":
    main()
