import asyncio
from datetime import datetime
import re
import sys
from typing import Sequence, TypedDict, cast
import polars as pl
from prisma import Prisma
from pydash import flatten, uniq
from prisma.models import FinancialSnapshot, Synonym
from prisma.types import (
    FinancialSnapshotCreateWithoutRelationsInput as FinancialSnapshotCreateInput,
)

from clients.finance.financials import CompanyFinancialExtractor
from clients.low_level.postgres.postgres import PsqlDatabaseClient
from constants.core import ETL_BASE_DATABASE_URL
from constants.company import COMPANY_INDICATORS
from data.etl.owner import OwnerEtl
from utils.re import get_or_re

FINANCIAL_KEYS = list(FinancialSnapshot.model_fields.keys())
ASSIGNEE_PATENT_THRESHOLD = 20


CompanyInfo = TypedDict("CompanyInfo", {"name": str, "symbol": str})


def transform_financials(
    records: Sequence[CompanyInfo], owner_map: dict[str, int]
) -> list[FinancialSnapshotCreateInput]:
    """
    Transform company rows

    - clean company name and attempts to match a synonym
    - looks up financial info
    """

    def fetch_financials(record: CompanyInfo):
        financials = CompanyFinancialExtractor(record["symbol"])
        return FinancialSnapshotCreateInput(
            owner_id=owner_map[record["name"].lower()],
            current_ratio=financials.current_ratio,
            debt_equity_ratio=financials.debt_equity_ratio,
            ebitda=financials.ebitda,
            gross_profit=financials.gross_profit,
            market_cap=financials.market_cap,
            net_debt=financials.net_debt,
            return_on_equity=financials.return_on_equity,
            return_on_research_capital=financials.return_on_research_capital,
            total_debt=financials.total_debt,
            snapshot_date=datetime.now(),
            symbol=record["symbol"],
        )

    return [fetch_financials(record) for record in records]


class AllOwnersEtl:
    @staticmethod
    async def get_owner_names():
        """
        Generates owner terms (assignee/inventor) from:
        - patent applications table
        - aact (ctgov)
        - drugcentral approvals
        """
        company_re = get_or_re(
            COMPANY_INDICATORS,
            enforce_word_boundaries=True,
            permit_plural=False,
            word_boundary_char=r"\y",
        )

        # attempts to select for companies & universities over individuals
        # (because the clustering makes a mess of individuals)
        db_owner_query_map = {
            # patents db
            "patents": f"""
                SELECT lower(unnest(assignees)) as name, count(*) as count
                FROM applications a
                GROUP BY name
                HAVING count(*) > {ASSIGNEE_PATENT_THRESHOLD} -- individuals unlikely to have more patents

                UNION ALL

                -- if fewer than 20 patents, BUT the name looks like a company, include it.
                SELECT max(name) as name, count(*) as count
                FROM applications a, unnest(assignees) as name
                where name ~* '{company_re}\\.?'
                GROUP BY lower(name)
                HAVING count(*) <= {ASSIGNEE_PATENT_THRESHOLD}

                UNION ALL

                SELECT lower(unnest(inventors)) as name, count(*) as count
                FROM applications a
                GROUP BY name
                HAVING count(*) > {ASSIGNEE_PATENT_THRESHOLD}

                UNION ALL

                SELECT lower(name) as name, count(*) as count
                FROM companies
                GROUP BY lower(name)
            """,
            # ctgov db
            "aact": """
                select lower(name) as name, count(*) as count
                from sponsors
                group by lower(name)
            """,
            # drugcentral db, with approvals
            # `ob_product`` has 1772 distinct applicants vs `approval` at 1136
            "drugcentral": """
                select lower(applicant) as name, count(*) as count
                from ob_product
                where applicant is not null
                group by lower(applicant)
            """,
        }
        rows = flatten(
            [
                await PsqlDatabaseClient(f"{ETL_BASE_DATABASE_URL}/{db}").select(query)
                for db, query in db_owner_query_map.items()
            ]
        )
        stock_names = [
            record["name"] for record in AllOwnersEtl.load_financial_company_info()
        ]
        names = uniq([row["name"] for row in rows]) + stock_names
        return names

    @staticmethod
    def load_financial_company_info() -> list[CompanyInfo]:
        nasdaq = pl.read_csv("data/NASDAQ.csv")
        nyse = pl.read_csv("data/NYSE.csv")
        all = nasdaq.vstack(nyse)
        pharmas = (
            all.filter(pl.col("Sector") == "Health Care")
            .with_columns(
                pl.col("Symbol").str.to_lowercase(),
                pl.col("Name").str.to_lowercase(),
            )
            .rename({"Symbol": "symbol", "Name": "name"})
        )
        return cast(
            list[CompanyInfo], pharmas.select(pl.col(["name", "symbol"])).to_dicts()
        )

    @staticmethod
    async def load_financials():
        """
        Data from https://www.nasdaq.com/market-activity/stocks/screener?exchange=NYSE
        """
        pharma_cos = AllOwnersEtl.load_financial_company_info()

        owner_recs = await Synonym.prisma().find_many(
            where={
                "AND": [
                    {"term": {"in": [record["name"] for record in pharma_cos]}},
                    {"owner_id": {"gt": 0}},  # returns non-null owner_id?
                ]
            },
        )
        owner_map = {
            record.term: record.owner_id
            for record in owner_recs
            if record.owner_id is not None
        }

        await FinancialSnapshot.prisma().create_many(
            data=transform_financials(pharma_cos, owner_map),
        )

    async def link_snapshots(self):
        """
        Link financial snapshots to owners
        """
        async with Prisma() as db:
            await db.execute_raw(
                f"""
                UPDATE owner
                SET financial_snapshot_id=synonym.owner_id
                FROM synonym
                WHERE owner.name=synonym.term
                AND synonym.owner_id IS NOT NULL;
                """
            )

    async def copy_all(self):
        names = await self.get_owner_names()
        await OwnerEtl().copy_all(names)
        await self.link_snapshots()


def main():
    asyncio.run(AllOwnersEtl().copy_all())


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.owner.load_owners
            """
        )
        sys.exit()

    main()
