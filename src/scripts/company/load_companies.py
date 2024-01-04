import asyncio
import re
import sys
import polars as pl
from clients.finance.financials import CompanyFinancials

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from constants.core import COMPANIES_TABLE_NAME
from typings.companies import Company, COMPANY_STR_KEYS


def clean_name(name: str) -> str:
    """
    Clean company name
    - removes mentions of stock type (e.g. common stock, preferred stock, etc.)
    """
    return re.sub(
        r"(?:class [a-z]? )?(?:(?:[0-9](?:\.[0-9]*)% )?(?:convertible |american )?(?:common|ordinary|preferred|voting|deposit[ao]ry) (?:stock|share|sh)s?|warrants?).*",
        "",
        name.lower(),
        flags=re.IGNORECASE,
    ).strip()


def transform_companies(rows, synonym_map) -> list[Company]:
    """
    Transform company rows

    - clean company name and attempts to match a synonym
    - looks up financial info
    """

    def transform_company(row):
        financials = CompanyFinancials(row["symbol"])
        return Company(
            id=row["id"],
            name=synonym_map.get(clean_name(row["name"]), clean_name(row["name"])),
            symbol=row["symbol"],
            current_ratio=financials.current_ratio,
            debt_equity_ratio=financials.debt_equity_ratio,
            ebitda=financials.ebitda,
            gross_profit=financials.gross_profit,
            is_bad_current_ratio=financials.is_bad_current_ratio,
            is_bad_debt_equity_ratio=financials.is_bad_debt_equity_ratio,
            is_low_return_on_equity=financials.is_low_return_on_equity,
            is_trading_below_cash=financials.is_trading_below_cash,
            is_troubled=financials.is_troubled,
            market_cap=financials.market_cap,
            net_debt=financials.net_debt,
            return_on_equity=financials.return_on_equity,
            return_on_research_capital=financials.return_on_research_capital,
            total_debt=financials.total_debt,
            synonyms=[row["name"], clean_name(row["name"])],
            parent_company_id=None,
        )

    return [transform_company(row) for row in rows]


async def load_companies():
    """
    Data from https://www.nasdaq.com/market-activity/stocks/screener?exchange=NYSE
    """
    nasdaq = pl.read_csv("data/NASDAQ.csv")
    nyse = pl.read_csv("data/NYSE.csv")
    all = nasdaq.vstack(nyse)
    pharmas = all.filter(pl.col("Sector") == "Health Care").select(
        pl.col(["Symbol", "Name"])
    )
    pharmas = pharmas.rename({col: col.lower() for col in pharmas.columns})
    pharmas = pharmas.with_columns(
        pl.col("symbol").alias("id"),
        *[
            pl.lit(None)
            .alias(col)
            .cast(pl.Boolean if col.startswith("is") else pl.Float32)
            for col in COMPANY_STR_KEYS
            if col not in ["id", "symbol", "parent_company_id", "synonyms"]
        ],
        pl.lit(None).alias("parent_company_id").cast(pl.Utf8),
        pl.lit(None).alias("synonyms"),
    )

    client = PsqlDatabaseClient()
    synonyms = await client.select("SELECT * FROM synonym_map")
    synonym_map = {synonym["synonym"]: synonym["term"] for synonym in synonyms}
    await client.create_and_insert(
        COMPANIES_TABLE_NAME,
        pharmas.to_dicts(),
        transform=lambda batch, _: transform_companies(batch, synonym_map),
    )


def main():
    asyncio.run(load_companies())


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.company.load_companies
            """
        )
        sys.exit()

    main()
