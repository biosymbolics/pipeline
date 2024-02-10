"""
Transformation methods for owner etl
"""

from datetime import datetime
from functools import reduce
import regex as re
from typing import Iterable, Sequence
from pydash import compact
from prisma.enums import OwnerType
from prisma.types import (
    FinancialSnapshotCreateWithoutRelationsInput as FinancialSnapshotCreateInput,
)


from clients.finance.financials import CompanyFinancialExtractor
from constants.company import (
    COMPANY_MAP,
    OWNER_SUPPRESSIONS,
    OWNER_TERM_NORMALIZATION_MAP,
)
from core.ner.classifier import classify_string
from core.ner.cleaning import RE_FLAGS
from core.ner.utils import cluster_terms
from typings.companies import CompanyInfo
from utils.re import get_or_re, sub_extra_spaces

from .constants import OWNER_KEYWORD_MAP


def clean_owners(
    owners: Sequence[str],
    owner_normalization_map: dict[str, str] = OWNER_TERM_NORMALIZATION_MAP,
    owner_suppressions: Sequence[str] = OWNER_SUPPRESSIONS,
) -> dict[str, str]:
    """
    Clean owner names
    - removes suppressions
    - removes 2x+ and trailing spaces
    - lower cases
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
        term_map_set = set(OWNER_TERM_NORMALIZATION_MAP.keys())

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
        lambda owners: [t.lower() for t in owners],
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


OwnerTypePriorityMap = {
    OwnerType.INDUSTRY_LARGE: 1,
    OwnerType.HEALTH_SYSTEM: 5,
    OwnerType.UNIVERSITY: 10,
    OwnerType.INDUSTRY: 20,
    OwnerType.GOVERNMENTAL: 30,
    OwnerType.FOUNDATION: 40,
    OwnerType.INDIVIDUAL: 50,
    OwnerType.OTHER_ORGANIZATION: 100,
    OwnerType.OTHER: 1000,
}


class OwnerTypeParser:
    @staticmethod
    def find(value: str) -> OwnerType:
        reason = sorted(
            classify_string(value, OWNER_KEYWORD_MAP, OwnerType.OTHER),
            key=lambda x: OwnerTypePriorityMap[x],
        )
        res = reason[0]
        return res
