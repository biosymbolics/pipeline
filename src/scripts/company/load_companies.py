import re
import sys
import polars as pl
from clients.finance.financials import CompanyFinancials

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from constants.core import COMPANIES_TABLE_NAME
from typings.companies import Company


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
            is_bad_current_ratio=financials.is_bad_current_ratio,
            is_bad_debt_equity_ratio=financials.is_bad_debt_equity_ratio,
            is_trading_below_cash=financials.is_trading_below_cash,
            is_troubled=financials.is_troubled,
            market_cap=financials.market_cap,
            net_debt=financials.net_debt,
            total_debt=financials.total_debt,
            synonyms=[row["name"], clean_name(row["name"])],
            parent_company_id=None,
        )

    return [transform_company(row) for row in rows]


def load_companies():
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
        pl.lit(None).alias("current_ratio").cast(pl.Float32),
        pl.lit(None).alias("debt_equity_ratio").cast(pl.Float32),
        pl.lit(None).alias("is_bad_debt_equity_ratio").cast(pl.Boolean),
        pl.lit(None).alias("is_bad_current_ratio").cast(pl.Boolean),
        pl.lit(None).alias("is_trading_below_cash").cast(pl.Boolean),
        pl.lit(None).alias("is_troubled").cast(pl.Boolean),
        pl.lit(None).alias("market_cap").cast(pl.Float32),
        pl.lit(None).alias("net_debt").cast(pl.Float32),
        pl.lit(None).alias("total_debt").cast(pl.Float32),
        pl.lit(None).alias("parent_company_id").cast(pl.Utf8),
        pl.lit(None).alias("synonyms"),
    )

    client = PsqlDatabaseClient()

    synonyms = client.select("SELECT * FROM synonym_map")
    synonym_map = {synonym["synonym"]: synonym["term"] for synonym in synonyms}
    client.create_and_insert(
        pharmas.to_dicts(),
        COMPANIES_TABLE_NAME,
        transform=lambda batch, _: transform_companies(batch, synonym_map),
    )


def main():
    load_companies()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.company.load_companies
            """
        )
        sys.exit()

    main()
