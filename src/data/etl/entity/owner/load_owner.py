import asyncio
import sys
import polars as pl
from pydash import flatten, uniq
from prisma.models import FinancialSnapshot

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from constants.core import ETL_BASE_DATABASE_URL
from constants.company import COMMON_GOVT_WORDS, COMMON_OWNER_WORDS
from typings.companies import CompanyInfo
from utils.re import get_or_re

from .owner import OwnerEtl


FINANCIAL_KEYS = list(FinancialSnapshot.model_fields.keys())
ASSIGNEE_PATENT_THRESHOLD = 50


class OwnerLoader:
    @staticmethod
    async def get_owners() -> tuple[list[str], list[int]]:
        """
        Generates owner terms (assignee/inventor) from:
        - patent applications table
        - aact (ctgov)
        - drugcentral approvals
        """
        org_re = get_or_re(
            [*COMMON_OWNER_WORDS, *COMMON_GOVT_WORDS],
            enforce_word_boundaries=True,
            permit_plural=True,
            word_boundary_char=r"\y",
        )

        # attempts to select for companies & universities over individuals
        # (because the clustering makes a mess of individuals)
        db_owner_query_map = {
            # patents db
            "patents": f"""
                -- if fewer than ASSIGNEE_PATENT_THRESHOLD, look for signs that it is an org
                SELECT lower(max(name)) as name, count(*) as count
                FROM applications a, unnest(assignees) as name
                WHERE
                    (
                        name ~* '{org_re}\\.?'
                        OR array_length(regexp_split_to_array(name, ' '), 1) > 5
                        OR array_length(regexp_split_to_array(name, ' '), 1) < 2
                    )
                GROUP BY lower(name)
                HAVING count(*) <= {ASSIGNEE_PATENT_THRESHOLD}

                UNION

                -- if greater than ASSIGNEE_PATENT_THRESHOLD, take it.
                SELECT lower(max(name)) as name, count(*) as count
                FROM applications a, unnest(assignees) as name
                GROUP BY lower(name)
                HAVING count(*) > {ASSIGNEE_PATENT_THRESHOLD}

                UNION

                -- if inventors are greater than ASSIGNEE_PATENT_THRESHOLD, take it.
                SELECT lower(max(name)) as name, count(*) as count
                FROM applications a, unnest(inventors) as name
                GROUP BY lower(name)
                HAVING count(*) > {ASSIGNEE_PATENT_THRESHOLD}
            """,
            # ctgov db
            "aact": """
                select lower(name) as name, (count(*) * 2) as count
                from sponsors
                group by lower(name)
            """,
            # drugcentral db, with approvals
            # `ob_product`` has 1772 distinct applicants vs `approval` at 1136
            "drugcentral": """
                select lower(applicant) as name, (count(*) * 20) as count
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
            record["name"].lower() for record in OwnerLoader.load_public_companies()
        ]
        names = [row["name"] for row in rows] + stock_names
        counts = [row["count"] for row in rows] + [1000] * len(stock_names)
        return names, counts

    @staticmethod
    def load_public_companies() -> list[CompanyInfo]:
        """
        Load public companies

        @source https://www.nasdaq.com/market-activity/stocks/screener
        """
        nasdaq = pl.read_csv("data/NASDAQ.csv")
        nyse = pl.read_csv("data/NYSE.csv")
        all = nasdaq.vstack(nyse)
        pharmas = (
            all.filter(pl.col("Sector").str.to_lowercase() == "health care")
            .rename({"Symbol": "symbol", "Name": "name"})
            .with_columns(
                pl.col("symbol").str.to_lowercase(),
                pl.col("name").str.to_lowercase(),
            )
        )
        return [
            CompanyInfo(**d)
            for d in pharmas.select(pl.col(["name", "symbol"])).to_dicts()
        ]

    async def copy_all(self, is_force_update: bool = False):
        names, counts = await self.get_owners()
        public_companies = self.load_public_companies()
        await OwnerEtl().copy_all(names, counts, public_companies, is_force_update)

    @staticmethod
    async def post_finalize():
        await OwnerEtl.post_finalize()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.entity.owner.load_owner [--post-finalize] [--force_update]
            """
        )
        sys.exit()

    if "--post-finalize" in sys.argv:
        asyncio.run(OwnerLoader().post_finalize())
    else:
        is_force_update = "--force_update" in sys.argv
        asyncio.run(OwnerLoader().copy_all(is_force_update))
