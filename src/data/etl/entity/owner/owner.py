"""
Class for biomedical entity etl
"""

import asyncio
from typing import Sequence
from pydash import flatten, group_by, omit, uniq
import logging
from prisma.models import FinancialSnapshot, Ownable, Owner, OwnerSynonym
from prisma.types import OwnerUpdateInput

from clients.low_level.prisma import batch_update, prisma_client
from clients.sec.sec_client import SecClient
from core.ner.cleaning import CleanFunction
from core.ner.normalizer import TermNormalizer
from data.etl.entity.base_entity_etl import BaseEntityEtl
from typings.companies import CompanyInfo
from typings.prisma import OwnerCreateWithSynonymsInput
from utils.classes import overrides
from utils.list import batch

from .constants import OwnerTypePriorityMap
from .transform import generate_clean_owner_map, OwnerTypeParser, transform_financials


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TYPE_FIELD = "default_type"


class OwnerEtl(BaseEntityEtl):
    """
    Class for owner etl

    - canonicalizes/normalizes terms
    - creates records for entities and corresponding relationships (e.g. parents, comprised_of)
    """

    def __init__(
        self,
        additional_cleaners: Sequence[CleanFunction] = [],
    ):
        self.normalizer = TermNormalizer(
            link=False,
            additional_cleaners=additional_cleaners,
        )

    def _generate_insert_records(
        self,
        names: Sequence[str],
        canonical_lookup_map: dict[str, str],
        ma_lookup_map: dict[str, int],
    ) -> list[OwnerCreateWithSynonymsInput]:
        """
        Create record dicts for entity insert

        Args:
        - names: list of names to insert
        - canonical_lookup_map: mapping of name to canonical name
        - ma_lookup_map: mapping of name to number of M&A filings
        """

        # merge records with same name
        flat_recs = [
            OwnerCreateWithSynonymsInput(
                name=(canonical_lookup_map.get(name) or name),
                owner_type=OwnerTypeParser().find(name),
                acquisition_count=ma_lookup_map.get(name) or 0,
                synonyms=[name],
            )
            for name in names
        ]
        grouped_recs = group_by(flat_recs, "name")
        merged_recs: list[OwnerCreateWithSynonymsInput] = [
            {
                **groups[0],
                "owner_type": (
                    sorted(
                        [g.get("owner_type") for g in groups],
                        key=lambda x: OwnerTypePriorityMap[x],
                    )
                )[0],
                "synonyms": uniq(flatten([(g.get("synonyms") or []) for g in groups])),
            }
            for groups in grouped_recs.values()
        ]

        return merged_recs

    async def create_records(self, names: Sequence[str], counts: Sequence[int]):
        """
        Create records for entities and relationships

        Args:
            names: list of names to insert
        """
        client = await prisma_client(600)
        canonical_lookup_map = generate_clean_owner_map(names, counts)
        ma_lookup_map = await self.generate_acquisition_map(
            uniq(canonical_lookup_map.values())
        )
        insert_recs = self._generate_insert_records(
            names, canonical_lookup_map, ma_lookup_map
        )

        # create flat records
        await Owner.prisma(client).create_many(
            data=[
                OwnerCreateWithSynonymsInput(**omit(ir, "synonyms"))  # type: ignore
                for ir in insert_recs
            ],
            skip_duplicates=True,
        )
        logger.info("Created owners")

        await batch_update(
            insert_recs,
            update_func=lambda r, tx: Owner.prisma(tx).update(
                where={"name": r["name"]},
                data=OwnerUpdateInput(
                    # todo: connectOrCreate when supported
                    synonyms={"create": [{"term": s} for s in uniq(r["synonyms"])]},
                ),
            ),
        )

        logger.info("Updated owners")

    @staticmethod
    async def load_financials(public_companies: Sequence[CompanyInfo]):
        """
        Data from https://www.nasdaq.com/market-activity/stocks/screener?exchange=NYSE
        """
        owner_recs = await OwnerSynonym.prisma().find_many(
            where={
                "AND": [
                    {"term": {"in": [co["name"] for co in public_companies]}},
                    {"owner_id": {"gt": 0}},  # filters out null owner_id
                ]
            },
        )
        owner_map = {
            record.term: record.owner_id
            for record in owner_recs
            if record.owner_id is not None
        }

        await FinancialSnapshot.prisma().create_many(
            data=transform_financials(public_companies, owner_map),
            skip_duplicates=True,
        )

    async def generate_acquisition_map(
        self, canonical_names: Sequence[str]
    ) -> dict[str, int]:
        """
        Load a simple map of company to number of M&A filings
        """
        sec_client = SecClient()

        batches = batch(canonical_names, 2000)

        async def handle_batch(b):
            filings = await sec_client.fetch_mergers_and_acquisitions(b)
            return {company: len(filings[company]) for company in b}

        batch_results = await asyncio.gather(*[handle_batch(b) for b in batches])
        return {k: v for d in batch_results for k, v in d.items()}

    async def delete_owners(self):
        """
        Delete all owners
        """
        logger.info("Deleting all owners")
        client = await prisma_client(600)
        await Ownable.prisma(client).query_raw(
            "UPDATE ownable SET owner_id=NULL WHERE owner_id IS NOT NULL"
        )
        await FinancialSnapshot.prisma(client).delete_many()
        await OwnerSynonym.prisma(client).delete_many()
        await Owner.prisma(client).delete_many()

    @overrides(BaseEntityEtl)
    async def copy_all(
        self,
        names: Sequence[str],
        counts: Sequence[int],
        public_companies: Sequence[CompanyInfo],
        is_update: bool = False,
    ):
        if is_update:
            await self.delete_owners()

        await self.create_records(names, counts)
        await self.load_financials(public_companies)

    @overrides(BaseEntityEtl)
    @staticmethod
    async def pre_finalize():
        pass

    @overrides(BaseEntityEtl)
    @staticmethod
    async def add_counts():
        """
        add counts to owner table (used for autocomplete ordering)
        """
        client = await prisma_client(600)
        await client.execute_raw("CREATE TEMP TABLE temp_count(id int, count int)")
        await client.execute_raw(
            f"INSERT INTO temp_count (id, count) SELECT owner_id as id, count(*) FROM ownable GROUP BY owner_id"
        )
        await client.execute_raw(
            "UPDATE owner ct SET count=temp_count.count FROM temp_count WHERE temp_count.id=ct.id"
        )
        await client.execute_raw("DROP TABLE IF EXISTS temp_count")

    @overrides(BaseEntityEtl)
    @staticmethod
    async def link_to_documents():
        """
        - Link "ownable" to canonical entities
        - add instance_rollup and category_rollups
        """
        client = await prisma_client(600)
        await client.execute_raw(
            f"""
            UPDATE ownable
            SET
                owner_id=owner_synonym.owner_id,
                canonical_name=owner.name,
                instance_rollup=owner.name,
                category_rollup=owner.name
            FROM owner_synonym, owner
            WHERE ownable.name=owner_synonym.term
            AND owner_synonym.owner_id=owner.id;
            """
        )

    @staticmethod
    async def _update_search_index():
        """
        update search index
        """
        client = await prisma_client(300)
        await client.execute_raw("DROP INDEX IF EXISTS owner_search_idx;")
        await client.execute_raw(
            f"""
            WITH synonym as (
                SELECT owner_id, array_agg(term) as terms
                FROM owner_synonym
                GROUP BY owner_id
            )
            UPDATE owner SET search = to_tsvector('english', name || ' ' || array_to_string(synonym.terms, ' '))
                FROM synonym
                WHERE owner_id=owner.id
                AND array_length(synonym.terms, 1) < 100000; -- dumb categorization error check
            """
        )
        await client.execute_raw(
            "CREATE INDEX owner_search_idx ON owner USING GIN(search);"
        )

    @staticmethod
    async def checksum():
        """
        Quick entity checksum
        """
        client = await prisma_client(300)
        checksums = {
            "owners": "SELECT COUNT(*) FROM owner",
            "ownable": "SELECT COUNT(*) FROM ownable",
            "empty_owner_id": "SELECT COUNT(*) FROM ownable WHERE owner_id IS NULL",
            "other_owner_id": "SELECT COUNT(*) FROM ownable, owner WHERE owner_id=owner.id AND owner.name='other'",
            "top_owners": "SELECT name, count FROM owner ORDER BY count DESC LIMIT 20",
            "financial_snapshots": "SELECT COUNT(*) FROM financials where owner_id IS NOT NULL",
        }
        results = await asyncio.gather(
            *[client.query_raw(query) for query in checksums.values()]
        )
        for key, result in zip(checksums.keys(), results):
            logger.warning(f"Load checksum {key}: {result}")
        return

    @staticmethod
    async def add_vectors():
        """
        Create company-level vectors that are the average of all patents
        (and other documents in the future)

        ivfflat "lists"  = rows / 1000 = 24183 / 1000 = 24
        """
        queries = [
            "DROP INDEX if exists owner_vector;",
            """
            UPDATE owner
            SET vector = agg.vector
            FROM (
                SELECT AVG(vector) as vector, ownable.owner_id as owner_id
                FROM patent, ownable
                WHERE patent.id=ownable.patent_id
                GROUP BY ownable.owner_id
            ) agg
            WHERE agg.owner_id=owner.id;
            """,
            "CREATE INDEX owner_vector ON owner USING ivfflat (vector vector_cosine_ops) WITH (lists = 24);",
        ]
        client = await prisma_client(600)

        for query in queries:
            await client.execute_raw(query)

    @overrides(BaseEntityEtl)
    @staticmethod
    async def post_finalize():
        """
        Run after:
            1) all biomedical entities are loaded
            2) all documents are loaded
            3) UMLS is loaded
        """
        await OwnerEtl.link_to_documents()
        await OwnerEtl.add_counts()
        await OwnerEtl._update_search_index()
        await OwnerEtl.add_vectors()
        await OwnerEtl.checksum()
