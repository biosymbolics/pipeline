"""
Class for biomedical entity etl
"""
from typing import Sequence
from prisma import Prisma
from pydash import flatten, group_by, omit, uniq
import logging
from prisma.models import FinancialSnapshot, Owner, OwnerSynonym
from prisma.types import OwnerUpdateInput

from clients.low_level.prisma import batch_update, prisma_client
from core.ner.cleaning import CleanFunction
from core.ner.normalizer import TermNormalizer
from data.etl.entity.base_entity_etl import BaseEntityEtl
from typings.companies import CompanyInfo
from typings.prisma import OwnerCreateWithSynonymsInput

from .transform import clean_owners, OwnerTypeParser, transform_financials


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TYPE_FIELD = "default_type"


class BaseOwnerEtl(BaseEntityEtl):
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
        lookup_map: dict[str, str],
    ) -> list[OwnerCreateWithSynonymsInput]:
        """
        Create record dicts for entity insert
        """

        # merge records with same name
        flat_recs = [
            OwnerCreateWithSynonymsInput(
                name=(lookup_map.get(name) or name),
                owner_type=OwnerTypeParser().find(name),
                synonyms=[name],
            )
            for name in names
        ]
        grouped_recs = group_by(flat_recs, "name")
        merged_recs: list[OwnerCreateWithSynonymsInput] = [
            {
                **groups[0],
                "synonyms": uniq(flatten([(g.get("synonyms") or []) for g in groups])),
            }
            for groups in grouped_recs.values()
        ]

        return merged_recs

    def generate_lookup_map(self, names: Sequence[str]) -> dict[str, str]:
        """
        Generate lookup map for names
        """
        return clean_owners(names)

    async def create_records(self, names: Sequence[str]):
        """
        Create records for entities and relationships

        Args:
            names: list of names to insert
        """
        client = await prisma_client(600)
        lookup_map = self.generate_lookup_map(names)
        insert_recs = self._generate_insert_records(names, lookup_map)

        # create flat records
        await Owner.prisma(client).create_many(
            data=[
                OwnerCreateWithSynonymsInput(**omit(ir, "synonyms"))  # type: ignore
                for ir in insert_recs
            ],
            skip_duplicates=True,
        )

        await batch_update(
            insert_recs,
            update_func=lambda r, tx: Owner.prisma(tx).update(
                where={"name": r["name"]},
                data=OwnerUpdateInput(
                    synonyms={"create": [{"term": s} for s in uniq(r["synonyms"])]},
                ),
            ),
        )

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

    async def copy_all(
        self, names: Sequence[str], public_companies: Sequence[CompanyInfo]
    ):
        await self.create_records(names)
        await self.load_financials(public_companies)

    @staticmethod
    async def pre_doc_finalize():
        pass

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
            "UPDATE biomedical_entity ct SET count=temp_count.count FROM temp_count WHERE temp_count.id=ct.id"
        )
        await client.execute_raw("DROP TABLE IF EXISTS temp_count")

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
                instance_rollup=biomedical_entity.name -- todo,
                category_rollup=biomedical_entity.name -- todo
            FROM owner_synonym, owner
            WHERE ownable.name=owner_synonym.term
            AND owner_synonym.owner_id=owner.id;
            """
        )

    @staticmethod
    async def post_doc_finalize():
        """
        Run after:
            1) all biomedical entities are loaded
            2) all documents are loaded
            3) UMLS is loaded
        """
        await BaseOwnerEtl.link_to_documents()
        await BaseOwnerEtl.add_counts()
