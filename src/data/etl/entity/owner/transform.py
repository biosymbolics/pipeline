"""
Transformation methods for owner etl
"""

from datetime import datetime
from functools import reduce
import regex as re
from typing import Iterable, Mapping, Sequence
from pydash import compact, group_by
from prisma.enums import OwnerType
from prisma.types import (
    FinancialSnapshotCreateWithoutRelationsInput as FinancialSnapshotCreateInput,
)
import logging


from clients.finance.financials import CompanyFinancialExtractor
from constants.company import (
    COMPANY_MAP,
    OWNER_SUPPRESSIONS,
    OWNER_TERM_NORMALIZATION_MAP,
    PLURAL_COMMON_OWNER_WORDS,
)
from core.ner.classifier import classify_string
from core.ner.cleaning import RE_FLAGS
from core.ner.utils import cluster_terms
from typings.companies import CompanyInfo
from utils.re import get_or_re, sub_extra_spaces

from .constants import OWNER_KEYWORD_MAP

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def generate_clean_owner_map(
    names: Sequence[str],
    counts: Sequence[int],
    normalization_map: dict[str, str] = OWNER_TERM_NORMALIZATION_MAP,
    overrides: dict[str, str] = COMPANY_MAP,
    suppressions: Sequence[str] = OWNER_SUPPRESSIONS,
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
        """
        other_res = [r"^the\b", r"\(.+\)"]
        other_re = get_or_re(other_res, enforce_word_boundaries=False, count="?")
        tail_removal = "[&., )(]+"

        suppress_re = (
            get_or_re(
                suppressions,
                enforce_word_boundaries=True,
                permit_plural=True,
                count="*",
            )
            + other_re
        )

        for term in terms:
            yield re.sub(suppress_re, "", term, flags=RE_FLAGS).strip(tail_removal)

    def normalize_terms(assignees: list[str]) -> Iterable[str]:
        """
        Normalize terms (e.g. "lab" -> "laboratory", "univ" -> "university")
        No deduplication, becaues the last step is tfidf + clustering
        """
        term_map_set = list(normalization_map.keys())
        term_map_re = get_or_re(term_map_set, enforce_word_boundaries=True)

        def _normalize(_assignee: str):
            assignee_copy = _assignee

            has_rewrites = (
                re.search(term_map_re, assignee_copy, flags=RE_FLAGS) is not None
            )
            if not has_rewrites:
                return assignee_copy

            for rewrite in term_map_set:
                assignee_copy = re.sub(
                    rf"\b{rewrite}\b",
                    f" {normalization_map[rewrite]} ",
                    assignee_copy,
                    flags=RE_FLAGS,
                ).strip()
            return assignee_copy

        for assignee in assignees:
            yield _normalize(assignee)

    def find_override(owner_str: str, key: str) -> str | None:
        """
        See if there is an explicit name mapping
        """
        has_mapping = re.search(rf"\b{key}\b", owner_str, flags=RE_FLAGS) is not None
        if has_mapping:
            return key
        return None

    def sub_punctuation(terms: list[str]) -> Iterable[str]:
        """
        Remove punctuation (.,)
        """
        for term in terms:
            yield re.sub(r"[.,]+", "", term, flags=RE_FLAGS)

    def generate_override_map(
        orig_names: Sequence[str], override_map: dict[str, str]
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

    logger.info("Generating owner canonicalization map (%s)", len(names))

    # apply overrides first
    override_map = generate_override_map(names, overrides)

    # clean the remainder
    cleaning_steps = [
        sub_suppressions,
        normalize_terms,  # order matters
        sub_punctuation,
        sub_extra_spaces,
        lambda owners: [o.lower() for o in owners],
    ]
    cleaned = list(reduce(lambda x, func: func(x), cleaning_steps, names))

    grouped = group_by(zip(cleaned, counts), lambda x: x[0])
    with_counts = {k: sum([x[1] for x in v]) for k, v in grouped.items()}

    # maps cleaned to clustered
    logger.info("Clustering owner terms (%s)", len(cleaned))
    cleaned_to_cluster = cluster_terms(with_counts, PLURAL_COMMON_OWNER_WORDS)

    logger.info("Generated owner canonicalization map")
    # e.g. { "Pfizer Inc": "pfizer", "Biogen Ma": "biogen" }
    return {
        orig: override_map.get(orig) or cleaned_to_cluster.get(clean) or "other"
        for orig, clean in zip(names, cleaned)
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
