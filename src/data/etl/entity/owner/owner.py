"""
Class for biomedical entity etl
"""

import asyncio
from datetime import datetime
from typing import Sequence
from pydash import flatten, group_by, omit, uniq
import logging
from prisma.models import Acquisition, FinancialSnapshot, Ownable, Owner, OwnerSynonym
from prisma.types import (
    AcquisitionCreateWithoutRelationsInput as AcquisitionCreateInput,
    OwnerUpdateInput,
)

from clients.low_level.prisma import batch_update, prisma_client
from clients.sec.sec_client import SecClient
from core.ner.cleaning import CleanFunction
from core.ner.normalizer import TermNormalizer
from data.etl.entity.base_entity_etl import BaseEntityEtl
from typings.companies import CompanyInfo
from typings.prisma import OwnerCreateWithSynonymsInput
from utils.classes import overrides

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
        term_normalizer: TermNormalizer,
    ):
        self.term_normalizer = term_normalizer

    @classmethod
    async def create(cls, additional_cleaners: Sequence[CleanFunction] = []):
        normalizer = await TermNormalizer.create(
            link=False,
            additional_cleaners=additional_cleaners,
        )
        return OwnerEtl(term_normalizer=normalizer)

    def _generate_insert_records(
        self,
        names: Sequence[str],
        canonical_lookup_map: dict[str, str],
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
        insert_recs = self._generate_insert_records(names, canonical_lookup_map)

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
        return

    async def load_financials(self, public_companies: Sequence[CompanyInfo]):
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

        sec_client = SecClient()
        symbol_map = {co["symbol"]: owner_map[co["name"]] for co in public_companies}
        filing_map = await sec_client.fetch_mergers_and_acquisitions(
            list(symbol_map.keys())
        )
        await Acquisition.prisma().create_many(
            data=[
                AcquisitionCreateInput(
                    owner_id=symbol_map[symbol],
                    accession_number=filing.accessionNo or "",
                    filing_date=datetime.fromisoformat(filing.filedAt),
                    url=filing.linkToTxt,
                )
                for symbol, filings in filing_map.items()
                for filing in filings
            ],
            skip_duplicates=True,
        )

    async def delete_owners(self):
        """
        Delete all owners
        """
        logger.info("Deleting all owners")
        client = await prisma_client(600)
        await Ownable.prisma(client).query_raw(
            "UPDATE ownable SET owner_id=NULL WHERE owner_id IS NOT NULL"
        )
        await Acquisition.prisma(client).delete_many()
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
        return

    @overrides(BaseEntityEtl)
    @staticmethod
    async def add_counts():
        """
        add counts to owner table (used for autocomplete ordering)
        """
        logger.info("Adding owner counts")
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
        logger.info("Linking owners")
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
            AND owner_synonym.owner_id=owner.id
            """
        )

        # hack!! avoid updating/inserting dups to being with
        # (though this will be ugly unless we map ownables -> owners prior to insert)
        await client.execute_raw(
            """
            DELETE FROM ownable o1
            USING ownable o2
            WHERE o1.owner_id=o2.owner_id
            AND o1.patent_id=o2.patent_id
            AND o1.patent_id IS NOT null
            AND o2.patent_id IS NOT null
            AND o1.id > o2.id;
            """
        )

    @staticmethod
    async def _update_search_index():
        """
        update search index
        """
        logger.info("Owner search index")
        client = await prisma_client(300)
        await client.execute_raw("DROP INDEX IF EXISTS owner_search")
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
        await client.execute_raw("CREATE INDEX owner_search ON owner USING GIN(search)")

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
        """
        logger.info("Owner vector and vector index")
        queries = [
            "DROP INDEX if exists owner_vector",
            """
            UPDATE owner
            SET vector = agg.vector
            FROM (
                SELECT AVG(vector) vector, owner_id
                FROM (
                    SELECT AVG(vector) vector, ownable.owner_id as owner_id
                    FROM ownable, trial
                    WHERE ownable.trial_id=trial.id
                    AND vector IS NOT null
                    AND owner_id IS NOT null
                    GROUP BY ownable.owner_id

                    UNION ALL

                    SELECT AVG(vector) vector, ownable.owner_id as owner_id
                    FROM ownable, patent
                    WHERE ownable.patent_id=patent.id
                    AND vector IS NOT null
                    AND owner_id IS NOT null
                    GROUP BY ownable.owner_id
                ) s
                GROUP BY owner_id
            ) agg
            WHERE agg.owner_id=owner.id
            """,
            "CREATE INDEX owner_vector ON owner USING hnsw (vector vector_cosine_ops)",
        ]
        client = await prisma_client(2400)

        for query in queries:
            await client.execute_raw(query)

    @overrides(BaseEntityEtl)
    @staticmethod
    async def finalize():
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
