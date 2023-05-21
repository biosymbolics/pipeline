"""
Workflows for building up sec data
"""

import asyncio
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


def __maybe_build_knowledge_graph(ticker, start_date):
    try:
        build_knowledge_graph(ticker, start_date)
    except Exception as ex:
        logging.error("failure to build kg for %s: %s", ticker, ex)


async def build_sec():
    start_date = datetime(2020, 1, 1)
    tasks = [
        asyncio.to_thread(__maybe_build_knowledge_graph, ticker, start_date)
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
