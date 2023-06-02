"""
Workflows for building up sec data
"""

import asyncio
from datetime import datetime
import logging

from sources.sec.build import build_indices

PHARMA_TICKERS = [
    # "PFE",
    # "JNJ",
    # "NVS",
    # "APPV",
    # "AMGN",
    # "GSK", # no 10k?
    # "GILD",
    # "NVO", # no 10k?
    # "TAK", # no 10k?
    # "LLY",
    # "AZN",
    # "BAYRY",
    # "RHHBY", # no 10k?
    # "MTZPY",
    # "MRK",
    "BMY",
]


def __build_indices(ticker, start_date):
    try:
        build_indices(ticker, start_date)
    except Exception as ex:
        logging.error("failure to build kg for %s: %s", ticker, ex)
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
    asyncio.run(main())
