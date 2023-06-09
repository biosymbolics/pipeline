"""
Workflows for building up sec data
"""

import asyncio
from datetime import datetime
import logging
import os
import traceback
from typing import Callable, Coroutine

from pydash import compact

from common.utils.async_utils import execute_async
from sources.sec.build import build_indices

logging.getLogger().setLevel(logging.INFO)

PHARMA_TICKERS = [
    "PFE",
    # "JNJ",
    # "NVS", # 20-F
    # "ABBV",
    # "AMGN",
    # "GSK", # 20-F
    # "GILD",
    # "NVO", # 20-F
    # "TAK", # 20-F
    # "LLY",
    # "AZN", # 20-F
    # "BAYRY", # 20-F
    # "RHHBY", # 20-F
    # "MTZPY", # 20-F
    # "MRK",
    # "BMY",
]


def __build_indices(
    ticker: str, start_date: datetime
) -> Callable[[], Coroutine[None, None, None]]:
    """
    Build indices closure

    Args:
        ticker (str): Ticker
        start_date (datetime): Start date
    """

    async def __build() -> None:
        try:
            build_indices(ticker, start_date)
        except Exception as ex:
            traceback.print_exc()
            logging.error("failure to build index for %s: %s", ticker, ex)

    return __build


async def build_sec():
    """
    Build SEC stuffs
    """
    start_date = datetime(2015, 1, 1)
    closures = [__build_indices(ticker, start_date) for ticker in PHARMA_TICKERS]
    await execute_async(compact(closures))


async def main():
    """
    Main
    """
    await build_sec()


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    asyncio.run(main())
