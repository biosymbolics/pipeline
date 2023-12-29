"""
Regulatory approvals client
"""
from functools import partial
import logging
import time
from typing import Sequence
from pydash import omit

from clients.low_level.boto3 import retrieve_with_cache_check, storage_decoder
from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import REGULATORY_APPROVAL_TABLE
from typings.approvals import RegulatoryApproval
from typings import QueryType, ApprovalSearchParams
from utils.sql import get_term_sql_query
from utils.string import get_id


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000

FIELDS = [
    # "applicant",
    "application_type",
    "approval_date",
    # "application_number",
    "brand_name",
    "generic_name",
    "indications",
    "label_url",
    "ndc_code",
    "pharmacologic_class",
    "pharmacologic_classes",
    "regulatory_agency",
]


def _search(
    terms: Sequence[str],
    query_type: QueryType = "AND",
    limit: int = MAX_SEARCH_RESULTS,
) -> list[RegulatoryApproval]:
    """
    Search regulatory approvals by terms
    """
    start = time.monotonic()

    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    term_query = get_term_sql_query(terms, query_type)

    query = f"""
        SELECT {", ".join(FIELDS)},
        ts_rank_cd(text_search, to_tsquery(%s)) AS score
        FROM {REGULATORY_APPROVAL_TABLE} as approvals
        WHERE text_search @@ to_tsquery(%s)
        AND lower(application_type) not in  ('otc monograph not final', 'otc monograph final', 'unapproved drug other')
        AND approval_date is not null
        ORDER BY score DESC
        LIMIT {limit}
    """

    results = PsqlDatabaseClient().select(query, [term_query, term_query])

    logger.info(
        "Search took %s seconds (%s)", round(time.monotonic() - start, 2), len(results)
    )

    approvals = [RegulatoryApproval(**omit(r, ["text_search"])) for r in results]

    return approvals


def search(p: ApprovalSearchParams) -> list[RegulatoryApproval]:
    """
    Search regulatory approvals by terms
    """
    args = {
        "terms": p.terms,
        "query_type": p.query_type,
    }
    key = get_id(
        {
            **args,
            "api": "approvals",
        }
    )
    search_partial = partial(_search, **args)

    if p.skip_cache == True:
        return search_partial(limit=p.limit)

    return retrieve_with_cache_check(
        search_partial,
        key=key,
        limit=p.limit,
        decode=lambda str_data: [
            RegulatoryApproval(**a) for a in storage_decoder(str_data)
        ],
    )