"""
Workflows for building up sec data
"""
import asyncio
from datetime import datetime
import logging
import sys
import traceback
from typing import Callable, Coroutine

from system import initialize

initialize()

from common.utils.async_utils import execute_async
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
    # "BIIB",
    "SNY",
    "VTRS",
    "REGN",
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


async def main():
    """
    Build SEC stuffs

    Usage:
        >>> python3 -m  workflows.enpv.build
    """
    start_date = datetime(2015, 1, 1)
    closures = [__build_indices(ticker, start_date) for ticker in PHARMA_TICKERS]
    await execute_async(closures)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
              Usage: python3 -m  workflows.enpv.build
              Load eNVP SEC data
        """
        )
        sys.exit()
    asyncio.run(main())
