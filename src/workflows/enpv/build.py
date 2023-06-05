"""
Workflows for building up sec data
"""

import asyncio
from datetime import datetime
import logging
import os

from sources.sec.build import build_indices

PHARMA_TICKERS = [
    "PFE",
    "JNJ",
    # "NVS", # 20-F
    "ABBV",
    "AMGN",
    # "GSK", # 20-F
    "GILD",
    # "NVO", # 20-F
    # "TAK", # 20-F
    "LLY",
    # "AZN", # 20-F
    # "BAYRY", # 20-F
    # "RHHBY", # 20-F
    # "MTZPY", # 20-F
    "MRK",
    "BMY",
]


def __build_indices(ticker, start_date):
    try:
        build_indices(ticker, start_date)
    except Exception as ex:
        logging.error("failure to build index for %s: %s", ticker, ex)
        raise ex


async def build_sec():
    """
    Build SEC stuffs
    """
    start_date = datetime(2022, 1, 1)
    tasks = [
        asyncio.to_thread(__build_indices, ticker, start_date)
        for ticker in PHARMA_TICKERS
    ]

    await asyncio.gather(*tasks)


async def main():
    """
    Main
    """
    await build_sec()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    asyncio.run(main())
