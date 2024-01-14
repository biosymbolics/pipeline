"""
Class for biomedical entity etl
"""
from prisma import Prisma
from typing import Sequence
from pydash import flatten, group_by, omit, uniq
import logging
from prisma.models import FinancialSnapshot, Owner, OwnerSynonym
from prisma.types import OwnerUpdateInput


from core.ner.cleaning import CleanFunction
from core.ner.normalizer import TermNormalizer
from data.etl.types import OwnerCreateWithSynonymsInput
from typings.companies import CompanyInfo

from .transform import clean_owners, OwnerTypeParser, transform_financials


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TYPE_FIELD = "default_type"


class BaseOwnerEtl:
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
        lookup_map = self.generate_lookup_map(names)
        insert_recs = self._generate_insert_records(names, lookup_map)

        # create flat records
        await Owner.prisma().create_many(
            data=[
                OwnerCreateWithSynonymsInput(**omit(ir, "synonyms"))  # type: ignore
                for ir in insert_recs
            ],
            skip_duplicates=True,
        )

        # update with synonym records
        for ir in insert_recs:
            await Owner.prisma().update(
                where={"name": ir["name"]},
                data=OwnerUpdateInput(
                    synonyms={"create": [{"term": s} for s in uniq(ir["synonyms"])]},
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
        db = Prisma(auto_register=True, http={"timeout": None})
        await db.connect()
        await self.create_records(names)
        await self.load_financials(public_companies)
        await db.disconnect()
