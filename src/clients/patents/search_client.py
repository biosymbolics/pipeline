"""
Patent client
"""
from functools import partial
import logging
import time
from typing import Sequence
from clients.companies.companies import get_company_map

from clients.low_level.boto3 import retrieve_with_cache_check
from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import (
    AGGREGATED_ANNOTATIONS_TABLE,
    ANNOTATIONS_TABLE,
    APPLICATIONS_TABLE,
    PATENT_TO_REGULATORY_APPROVAL_TABLE,
    PUBLICATION_NUMBER_MAP_TABLE,
    REGULATORY_APPROVAL_TABLE,
    PATENT_TO_TRIAL_TABLE,
    TRIALS_TABLE,
)
from typings.patents import ScoredPatentApplication as PatentApplication
from typings import PatentSearchParams, QueryType, TermField
from utils.string import get_id

from .enrich import enrich_search_result
from .types import QueryPieces
from .utils import get_max_priority_date


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_SEARCH_RESULTS = 2000
MAX_ARRAY_LENGTH = 50
EXEMPLAR_SIMILARITY_THRESHOLD = 0.7

"""
Larger decay rates will result in more matches

Usage:
    EXP(-annotation.character_offset_start * {DECAY_RATE}) > {threshold})
"""
DECAY_RATE = 1 / 2000


SEARCH_RETURN_FIELDS = {
    "apps.publication_number": "publication_number",
    "apps.title": "title",
    "abstract": "abstract",
    "application_number": "application_number",
    "country": "country",
    "domains": "domains",
    "embeddings::real[]": "embeddings",
    "family_id": "family_id",
    "ipc_codes": "ipc_codes",
    "priority_date": "priority_date",
    "similar_patents": "similar_patents",
    "url": "url",
}

APPROVED_SEARCH_RETURN_FIELDS = {
    "max(approval_indications)": "approval_indications",
    "max(brand_name)": "brand_name",
    "max(generic_name)": "generic_name",
}

TRIAL_RETURN_FIELDS = {
    "array_agg(distinct trials.nct_id)": "nct_ids",
    "(array_agg(trials.phase ORDER BY trials.start_date desc))[1]": "max_trial_phase",
    "(array_agg(trials.status ORDER BY trials.start_date desc))[1]": "last_trial_status",
    "(array_agg(trials.last_updated_date ORDER BY trials.start_date desc))[1]": "last_trial_update",
    "(array_agg(trials.termination_reason ORDER BY trials.start_date desc))[1]": "termination_reason",
}

FIELDS: list[str] = [
    *[
        f"{field} as {new_field}"
        for field, new_field in {
            **SEARCH_RETURN_FIELDS,
            **APPROVED_SEARCH_RETURN_FIELDS,
        }.items()
    ],
    *[f"{field} as {new_field}" for field, new_field in TRIAL_RETURN_FIELDS.items()],
]


def __get_query_pieces(
    terms: Sequence[str],
    exemplar_embeddings: Sequence[str],
    query_type: QueryType,
    min_patent_years: int,
) -> QueryPieces:
    """
    Helper to generate pieces of patent search query
    """
    is_id_search = any([t.startswith("WO-") for t in terms])
    lower_terms = [t.lower() for t in terms]

    # if ids, ignore most of the standard criteria
    if is_id_search:
        if (
            not all([t.startswith("WO-") for t in terms])
            or len(exemplar_embeddings) > 0
        ):
            raise ValueError("Cannot mix id and (term or exemplar patent) search")

        return QueryPieces(
            fields=[*FIELDS, "1 as search_rank", "0 as exemplar_similarity"],
            where=f"WHERE apps.publication_number = ANY(%s)",
            params=[terms],
            cosine_source="",
        )

    # exp decay scaling for search terms; higher is better
    fields = [
        *FIELDS,
        f"AVG(EXP(-annotation.character_offset_start * {DECAY_RATE})) as search_rank",
    ]
    max_priority_date = get_max_priority_date(int(min_patent_years))
    date_criteria = f"priority_date > '{max_priority_date}'::date"

    # search term comparison
    comparison = "&&" if query_type == "OR" else "@>"
    term_criteria = f"search_terms {comparison} %s"

    if len(exemplar_embeddings) > 0:
        exemplar_criterion = [
            f"(1 - (embeddings <=> '{e}')) > {EXEMPLAR_SIMILARITY_THRESHOLD}"
            for e in exemplar_embeddings
        ]
        exemplar_criteria = f"AND ({f' {query_type} '.join(exemplar_criterion)})"
        cosine_scores = [f"(1 - (embeddings <=> '{e}'))" for e in exemplar_embeddings]
        cosine_source = f", unnest (ARRAY[{','.join(cosine_scores)}]) cosine_scores"
        fields.append("AVG(cosine_scores) as exemplar_similarity")
    else:
        exemplar_criteria = ""
        cosine_source = ""
        fields.append("0 as exemplar_similarity")

    return QueryPieces(
        fields=fields,
        where=f"WHERE {date_criteria} AND {term_criteria} {exemplar_criteria}",
        params=[lower_terms],
        cosine_source=cosine_source,
    )


def _get_exemplar_embeddings(exemplar_patents: Sequence[str]) -> list[str]:
    """
    Get embeddings for exemplar patents
    """
    return [
        rec["embeddings"]
        for rec in PsqlDatabaseClient().select(
            "SELECT embeddings FROM applications WHERE publication_number = ANY(%s)",
            [exemplar_patents],
        )
    ]


def _search(
    terms: Sequence[str],
    exemplar_patents: Sequence[str] = [],
    query_type: QueryType = "AND",
    min_patent_years: int = 10,
    term_field: TermField = "terms",
    limit: int = MAX_SEARCH_RESULTS,
) -> list[PatentApplication]:
    """
    Search patents by terms
    """
    start = time.monotonic()

    if not isinstance(terms, list):
        logger.error("Terms must be a list: %s (%s)", terms, type(terms))
        raise ValueError("Terms must be a list")

    exemplar_embeddings = (
        _get_exemplar_embeddings(exemplar_patents) if len(exemplar_patents) > 0 else []
    )
    qp = __get_query_pieces(terms, exemplar_embeddings, query_type, min_patent_years)

    query = f"""
        SELECT {", ".join(qp["fields"])},
        max(agg_annotations.{term_field}) as terms,
        (CASE
            WHEN max(approval_dates) IS NOT NULL AND ARRAY_LENGTH(max(approval_dates), 1) > 0
            THEN (max(approval_dates))[1]
            ELSE NULL END
        ) as approval_date,
        (CASE WHEN max(approval_dates) IS NOT NULL THEN True ELSE False END) as is_approved
        FROM {APPLICATIONS_TABLE} AS apps
        JOIN {AGGREGATED_ANNOTATIONS_TABLE} as agg_annotations ON (agg_annotations.publication_number = apps.publication_number)
        JOIN {ANNOTATIONS_TABLE} as annotation ON annotation.publication_number = apps.publication_number -- for search_rank
        JOIN {PUBLICATION_NUMBER_MAP_TABLE} as pub_map ON pub_map.publication_number = apps.publication_number
        LEFT JOIN {PATENT_TO_REGULATORY_APPROVAL_TABLE} p2a ON p2a.publication_number = pub_map.other_publication_number
        LEFT JOIN {REGULATORY_APPROVAL_TABLE} approvals ON approvals.regulatory_application_number = p2a.regulatory_application_number
        LEFT JOIN {PATENT_TO_TRIAL_TABLE} a2t ON a2t.publication_number = apps.publication_number
        LEFT JOIN {TRIALS_TABLE} ON trials.nct_id = a2t.nct_id
        {qp["cosine_source"]}
        {qp["where"]}
        GROUP BY {",".join(SEARCH_RETURN_FIELDS.keys())}
        ORDER BY priority_date desc
        LIMIT {limit}
    """

    results = PsqlDatabaseClient().select(query, qp["params"])

    company_map = get_company_map()
    enriched_results = enrich_search_result(results, company_map)

    logger.info(
        "Search took %s seconds (%s)", round(time.monotonic() - start, 2), len(results)
    )

    return enriched_results


def search(p: PatentSearchParams) -> list[PatentApplication]:
    """
    Search patents by terms
    Filters on
    - lower'd terms
    - priority date
    - at least one composition of matter IPC code

    Args:
        p.terms (Sequence[str]): list of terms to search for
        p.exemplar_patents (Sequence[str], optional): list of exemplar patents to search for. Defaults to [].
        p.query_type (QueryType, optional): whether to search for patents with all terms (AND) or any term (OR). Defaults to "AND".
        p.min_patent_years (int, optional): minimum patent age in years. Defaults to 10.
        p.term_field (TermField, optional): which field to search on. Defaults to "terms".
                Other values are `instance_rollup` (which are rollup terms at a high level of specificity, e.g. "Aspirin 50mg" might have a rollup term of "Aspirin")
                and `category_rollup` (wherein "Aspirin 50mg" might have a rollup category of "NSAIDs")
        p.limit (int, optional): max results to return. Defaults to MAX_SEARCH_RESULTS.
        p.skip_cache (bool, optional): whether to skip cache. Defaults to False.

    Returns: a list of matching patent applications

    Example:
    ```
    from clients.patents import search_client
    from handlers.patents.types import PatentSearchParams
    p = search_client.search(PatentSearchParams(terms=['migraine disorders'], skip_cache=True, limit=5))
    [t.search_rank for t in p]
    ```
    """
    args = {
        "terms": p.terms,
        "exemplar_patents": p.exemplar_patents,
        "query_type": p.query_type,
        "min_patent_years": p.min_patent_years,
        "term_field": p.term_field,
    }
    key = get_id(args)
    search_partial = partial(_search, **args)

    if p.skip_cache == False:
        return search_partial(limit=p.limit)

    return retrieve_with_cache_check(search_partial, key=key, limit=p.limit)
