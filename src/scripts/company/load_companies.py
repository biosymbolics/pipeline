import re
import sys
import polars as pl

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from constants.core import COMPANIES_TABLE_NAME
from typings.companies import Company


def clean_name(name: str) -> str:
    return re.sub(
        r"(?:class [a-z]? )?(?:(?:[0-9](?:\.[0-9]*)% )?(?:convertible |american )?(?:common|ordinary|preferred|voting|deposit[ao]ry) (?:stock|share|sh)s?|warrants?).*",
        "",
        name.lower(),
        flags=re.IGNORECASE,
    ).strip()


def transform_company(rows, synonym_map) -> list[Company]:
    return [
        Company(
            id=row["id"],
            name=synonym_map.get(clean_name(row["name"]), clean_name(row["name"])),
            symbol=row["symbol"],
            market_cap=row["market_cap"],
            net_debt=row["net_debt"],
            is_trading_below_cash=row["is_trading_below_cash"],
            synonyms=[row["name"], clean_name(row["name"])],
            parent_company_id=None,
        )
        for row in rows
    ]


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
        pl.lit(False).alias("is_trading_below_cash"),
        pl.lit(None).alias("market_cap"),
        pl.lit(None).alias("net_debt"),
        pl.lit(None).alias("parent_company_id"),
        pl.lit(None).alias("synonyms"),
    )

    client = PsqlDatabaseClient()

    synonyms = client.select("SELECT * FROM synonym_map")
    synonym_map = {synonym["synonym"]: synonym["term"] for synonym in synonyms}
    client.create_and_insert(
        pharmas.to_dicts(),
        COMPANIES_TABLE_NAME,
        transform=lambda batch, _: transform_company(batch, synonym_map),
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
