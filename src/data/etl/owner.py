"""
Class for biomedical entity etl
"""
from functools import reduce
from prisma import Prisma
import regex as re
from typing import Iterable, Sequence, cast
from prisma.enums import OwnerType
from prisma.models import Owner
from prisma.types import OwnerUpdateInput
from pydash import flatten, group_by, omit, uniq
import logging

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
from utils.re import get_or_re, remove_extra_spaces


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_TYPE_FIELD = "default_type"


def clean_owners(owners: Sequence[str]) -> dict[str, str]:
    """
    Clean owner names
    - removes suppressions
    - removes 2x+ and trailing spaces
    - title cases
    - normalizes terms (e.g. "lab" -> "laboratory", "univ" -> "university")
    - applies overrides (e.g. "biogen ma" -> "biogen")
    - then does clustering
    """

    def remove_suppressions(terms: list[str]) -> Iterable[str]:
        """
        Remove term suppressions (generic terms like LLC, country, etc)
        Examples:
            - Matsushita Electric Ind Co Ltd -> Matsushita
            - MEDIMMUNE LLC -> Medimmune
            - University Of Alabama At Birmingham  -> University Of Alabama
            - University of Colorado, Denver -> University of Colorado
        """
        post_suppress_re = rf"(?:(?:\s+|,){get_or_re(OWNER_SUPPRESSIONS)}\b)"
        pre_suppress_re = "^the"
        suppress_re = rf"(?:{pre_suppress_re}|{post_suppress_re}|((?:,|at) .*$)|\(.+\))"

        for term in terms:
            yield re.sub(suppress_re, "", term, flags=RE_FLAGS).rstrip("&[ .,]*")

    def normalize_terms(assignees: list[str]) -> Iterable[str]:
        """
        Normalize terms (e.g. "lab" -> "laboratory", "univ" -> "university")
        """
        term_map_set = set(OWNER_TERM_MAP.keys())

        def _normalize(cleaned: str):
            terms_to_rewrite = list(term_map_set.intersection(cleaned.lower().split()))
            if len(terms_to_rewrite) > 0:
                _assignee = assignee
                for term in terms_to_rewrite:
                    _assignee = re.sub(
                        rf"\b{term}\b",
                        f" {OWNER_TERM_MAP[term.lower()]} ",
                        cleaned,
                        flags=RE_FLAGS,
                    ).strip()
                return _assignee

            return assignee

        for assignee in assignees:
            yield _normalize(assignee)

    def get_lookup_mapping(
        clean_assignee: str, og_assignee: str, key: str
    ) -> str | None:
        """
        See if there is an explicit name mapping on cleaned or original assignee
        """
        to_check = [clean_assignee, og_assignee]
        has_mapping = any(
            [
                re.match(rf"\b{key}\b", check, flags=RE_FLAGS) is not None
                for check in to_check
            ]
        )
        if has_mapping:
            return key
        return None

    def apply_overrides(
        assignees: list[str], cleaned_orig_map: dict[str, str]
    ) -> Iterable[str]:
        def _map(cleaned: str):
            og_assignee = cleaned_orig_map[cleaned]
            mappings = [
                key
                for key in COMPANY_MAP.keys()
                if get_lookup_mapping(cleaned, og_assignee, key)
            ]
            if len(mappings) > 0:
                logging.debug(
                    "Found mapping for assignee: %s -> %s", assignee, mappings[0]
                )
                return COMPANY_MAP[mappings[0]]
            return assignee

        for assignee in assignees:
            yield _map(assignee)

    def title(assignees: list[str]) -> Iterable[str]:
        for assignee in assignees:
            yield assignee.title()

    cleaning_steps = [
        remove_suppressions,
        normalize_terms,  # order matters
        remove_extra_spaces,
        title,
    ]
    cleaned = list(reduce(lambda x, func: func(x), cleaning_steps, owners))
    cleaned_orig_map = dict(zip(cleaned, owners))

    with_overiddes = list(apply_overrides(cleaned, cleaned_orig_map))

    return cast(dict[str, str], cluster_terms(with_overiddes, return_dict=True))


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
            "NYU",
            "Universitaire?s?",
            # "l'Université",
            # "Université",
            "Universita(?:ri)?",
            "education",
            "Universidad",
        ],
        OwnerType.INDUSTRY_LARGE: LARGE_PHARMA_KEYWORDS,
        OwnerType.INDUSTRY: [
            *COMPANY_STRINGS,
            "laboratories",
            "Procter and Gamble",
            "3M",
            "Neuroscience$",
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
            "NIH",
            "VA",
            "European Organisation",
            "EORTC",
            "Assistance Publique",
            "FDA",
            "Bureau",
            "Authority",
        ],
        OwnerType.HEALTH_SYSTEM: [
            "healthcare",
            "(?:medical|cancer|health) (?:center|centre|system|hospital)s?",
            "clinics?",
            "districts?",
        ],
        OwnerType.FOUNDATION: ["foundatations?", "trusts?"],
        OwnerType.OTHER_ORGANIZATION: [
            "Research Network",
            "Alliance",
            "Group$",
            "research cent(?:er|re)s?",
        ],
        OwnerType.INDIVIDUAL: [r"M\.?D\.?"],
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
        entity_recs = self._generate_insert_records(names, lookup_map)

        # create flat records
        await Owner.prisma().create_many(
            data=[
                OwnerCreateWithSynonymsInput(**omit(er, "synonyms"))  # type: ignore
                for er in entity_recs
            ],
            skip_duplicates=True,
        )

        # update with synonym records
        for er in entity_recs:
            await Owner.prisma().update(
                where={"name": er["name"]},
                data=OwnerUpdateInput(
                    synonyms={"create": [{"term": s} for s in er["synonyms"]]},
                ),
            )

    async def copy_all(self, names: Sequence[str]):
        db = Prisma(auto_register=True, http={"timeout": None})
        await db.connect()
        await self.create_records(names)
        await db.disconnect()
