"""
Workflows for building up data
"""

from datetime import datetime
import logging

from sources.sec.knowledge_graph import build_knowledge_graph

PHARMA_TICKERS = [
    "PFE",
    "JNJ",
    "NVS",
    "APPV",
    "AMGN",
    "GSK",
    "GILD",
    "NVO",
    "TAK",
    "LLY",
    "AZN",
    "BAYRY",
    "RHHBY",
    "MTZPY",
    "MRK",
    "BMY",
]


def build_sec():
    start_date = datetime(2020, 1, 1)
    for ticker in PHARMA_TICKERS:
        try:
            build_knowledge_graph(ticker, start_date)
        except Exception as ex:
            logging.error("Error encountered building %s: %s", ticker, ex)


def main():
    """
    Main
    """
    build_sec()


if __name__ == "__main__":
    main()
