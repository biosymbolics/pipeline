"""
Class for biomedical entity etl
"""
from datetime import datetime
from functools import reduce
from prisma import Prisma
import regex as re
from typing import Iterable, Sequence
from prisma.enums import OwnerType
from prisma.models import FinancialSnapshot, Owner, OwnerSynonym
from prisma.types import (
    OwnerUpdateInput,
    FinancialSnapshotCreateWithoutRelationsInput as FinancialSnapshotCreateInput,
)

from pydash import compact, flatten, group_by, omit, uniq
import logging

from clients.finance.financials import CompanyFinancialExtractor
from constants.company import COMPANY_STRINGS, LARGE_PHARMA_KEYWORDS
from constants.company import (
    COMPANY_MAP,
    OWNER_SUPPRESSIONS,
    OWNER_TERM_MAP,
)
from core.ner.classifier import classify_string, create_lookup_map
from core.ner.cleaning import RE_FLAGS, CleanFunction
from core.ner.normalizer import TermNormalizer
from core.ner.utils import cluster_terms
from data.etl.types import OwnerCreateWithSynonymsInput
from typings.companies import CompanyInfo
from utils.re import get_or_re, sub_extra_spaces


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TYPE_FIELD = "default_type"


def clean_owners(
    owners: Sequence[str],
    owner_normalization_map: dict[str, str] = OWNER_TERM_MAP,
    owner_suppressions: Sequence[str] = OWNER_SUPPRESSIONS,
) -> dict[str, str]:
    """
    Clean owner names
    - removes suppressions
    - removes 2x+ and trailing spaces
    - title cases
    - normalizes terms (e.g. "lab" -> "laboratory", "univ" -> "university")
    - applies overrides (e.g. "biogen ma" -> "biogen")
    - then does clustering
    """

    def sub_suppressions(terms: list[str]) -> Iterable[str]:
        """
        Remove term suppressions (generic terms like LLC, country, etc)
        Examples:
            - Matsushita Electric Ind Co Ltd -> Matsushita
            - MEDIMMUNE LLC -> Medimmune
            - University Of Alabama At Birmingham  -> University Of Alabama
            - University of Colorado, Denver -> University of Colorado
        """
        post_suppress_re = rf"(?:(?:\s+|,){get_or_re(owner_suppressions)}\b)"
        pre_suppress_re = "^the"
        suppress_re = rf"(?:{pre_suppress_re}|{post_suppress_re}|((?:,|at) .*$)|\(.+\))"

        for term in terms:
            yield re.sub(suppress_re, "", term, flags=RE_FLAGS).rstrip("&[ .,]*")

    def normalize_terms(assignees: list[str]) -> Iterable[str]:
        """
        Normalize terms (e.g. "lab" -> "laboratory", "univ" -> "university")
        No deduplication, becaues the last step is tfidf + clustering
        """
        term_map_set = set(OWNER_TERM_MAP.keys())

        def _normalize(cleaned: str):
            terms_to_rewrite = list(term_map_set.intersection(cleaned.lower().split()))
            if len(terms_to_rewrite) > 0:
                _assignee = assignee
                for term in terms_to_rewrite:
                    _assignee = re.sub(
                        rf"\b{term}\b",
                        f" {owner_normalization_map[term.lower()]} ",
                        cleaned,
                        flags=RE_FLAGS,
                    ).strip()
                return _assignee

            return assignee

        for assignee in assignees:
            yield _normalize(assignee)

    def find_override(owner_str: str, key: str) -> str | None:
        """
        See if there is an explicit name mapping
        """
        has_mapping = re.match(rf"\b{key}\b", owner_str, flags=RE_FLAGS) is not None
        if has_mapping:
            return key
        return None

    def generate_override_map(
        orig_names: Sequence[str], override_map: dict[str, str] = COMPANY_MAP
    ) -> dict[str, str]:
        """
        Apply overrides to owner strings, a mapping between original_name and overridden name
        e.g. {"pfizer inc": "pfizer"}
        """

        def _map(orig_name: str):
            # find (first) key of match (e.g. "johns? hopkins?"), if exists
            override_key = next(
                filter(
                    lambda matching_str: find_override(orig_name, matching_str),
                    override_map.keys(),
                ),
                None,
            )
            return (orig_name, override_map[override_key]) if override_key else None

        return dict(compact([_map(on) for on in orig_names]))

    # apply overrides first
    override_map = generate_override_map(owners)

    # clean the remainder
    cleaning_steps = [
        sub_suppressions,
        normalize_terms,  # order matters
        sub_extra_spaces,
        lambda owners: [t.title() for t in owners],
    ]
    cleaned = list(reduce(lambda x, func: func(x), cleaning_steps, owners))

    # maps orig to cleaned
    orig_to_cleaned = {
        **{orig: clean for orig, clean in zip(owners, cleaned)},
        **override_map,
    }

    # maps cleaned to clustered
    cleaned_to_cluster = cluster_terms(list(orig_to_cleaned.values()))

    return {
        orig: cleaned_to_cluster.get(clean) or clean
        for orig, clean in orig_to_cleaned.items()
    }


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


class OwnerTypeParser:
    @staticmethod
    def find(value: str) -> OwnerType:
        reason = classify_string(value, OWNER_KEYWORD_MAP, OwnerType.OTHER)
        res = reason[0]
        return res


OWNER_KEYWORD_MAP = create_lookup_map(
    {
        OwnerType.UNIVERSITY: [
            "univ(?:ersit(?:y|ies))?",
            "colleges?",
            "research hospitals?",
            "institute?s?",
            "schools?",
            "nyu",
            "universitaire?s?",
            # "l'Université",
            # "Université",
            "universita(?:ri)?",
            "education",
            "universidad",
        ],
        OwnerType.INDUSTRY_LARGE: LARGE_PHARMA_KEYWORDS,
        OwnerType.INDUSTRY: [
            *COMPANY_STRINGS,
            "laboratories",
            "procter and gamble",
            "3m",
            "neuroscience$",
            "associates",
            "medical$",
        ],
        OwnerType.GOVERNMENTAL: [
            "government",
            "govt",
            "federal",
            "national",
            "state",
            "us health",
            "veterans affairs",
            "nih",
            "va",
            "european organisation",
            "eortc",
            "assistance publique",
            "fda",
            "bureau",
            "authority",
        ],
        OwnerType.HEALTH_SYSTEM: [
            "healthcare",
            "(?:medical|cancer|health) (?:center|centre|system|hospital)s?",
            "clinics?",
            "districts?",
        ],
        OwnerType.FOUNDATION: ["foundatations?", "trusts?"],
        OwnerType.OTHER_ORGANIZATION: [
            "research network",
            "alliance",
            "group$",
            "research cent(?:er|re)s?",
        ],
        OwnerType.INDIVIDUAL: [r"m\.?d\.?", "dr\\.?", "ph\\.?d\\.?"],
    }
)

ASSIGNEE_PATENT_THRESHOLD = 20


class OwnerEtl:
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
