import asyncio
import sys
import polars as pl
from pydash import flatten, uniq
from prisma.models import FinancialSnapshot

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from constants.core import ETL_BASE_DATABASE_URL
from constants.company import COMPANY_INDICATORS
from typings.companies import CompanyInfo
from utils.re import get_or_re

from .owner import BaseOwnerEtl


FINANCIAL_KEYS = list(FinancialSnapshot.model_fields.keys())
ASSIGNEE_PATENT_THRESHOLD = 20


class OwnerLoader:
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
                SELECT lower(max(name)) as name
                FROM applications a, unnest(assignees) as name
                GROUP BY lower(name)
                HAVING count(*) > {ASSIGNEE_PATENT_THRESHOLD} -- individuals unlikely to have more patents

                UNION ALL

                -- if fewer than 20 patents, BUT the name looks like a company, include it.
                SELECT lower(max(name)) as name
                FROM applications a, unnest(assignees) as name
                WHERE name ~* '{company_re}\\.?'
                GROUP BY lower(name)
                HAVING count(*) <= {ASSIGNEE_PATENT_THRESHOLD}

                UNION ALL

                SELECT lower(max(name)) as name
                FROM applications a, unnest(inventors) as name
                GROUP BY lower(name)
                HAVING count(*) > {ASSIGNEE_PATENT_THRESHOLD}
            """,
            # ctgov db
            "aact": """
                select lower(name) as name
                from sponsors
                group by lower(name)
            """,
            # drugcentral db, with approvals
            # `ob_product`` has 1772 distinct applicants vs `approval` at 1136
            "drugcentral": """
                select lower(applicant) as name
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
        stock_names = [record["name"] for record in OwnerLoader.load_public_companies()]
        names = uniq([row["name"] for row in rows]) + stock_names
        return names

    @staticmethod
    def load_public_companies() -> list[CompanyInfo]:
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

    async def copy_all(self):
        names = await self.get_owner_names()
        public_companies = self.load_public_companies()
        await BaseOwnerEtl().copy_all(names, public_companies)


def main():
    asyncio.run(OwnerLoader().copy_all())


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.entity.owner.load
            """
        )
        sys.exit()

    main()
