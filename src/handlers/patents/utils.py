import json

from clients.patents.types import QueryType, TermField, get_query_type
from handlers.patents.types import (
    OptionalPatentSearchParams,
    PatentSearchParams,
    ParsedPatentSearchParams,
)


def parse_bool(value: bool | str | None) -> bool:
    if value is None:
        return False
    return json.loads(str(value).lower())


def parse_params(
    _params: PatentSearchParams,
    default_params: OptionalPatentSearchParams = {},
    default_limit: int = 1000,
) -> ParsedPatentSearchParams:
    """
    Parse patent params
    """
    # combine default and provided params
    params: PatentSearchParams = {**default_params, **_params}

    # parse ";"-delimited terms
    terms = params.get("terms")
    terms_list = [t.strip() for t in (terms.split(";") if terms else [])]

    is_exhaustive = parse_bool(params.get("is_exhaustive", "false"))
    limit = params.get("limit") or default_limit
    min_patent_years = params.get("min_patent_years") or 10
    query_type: QueryType = get_query_type(params.get("query_type"))
    skip_cache = parse_bool(params.get("skip_cache", "false"))
    term_field: TermField = params.get("term_field") or "terms"

    return {
        "terms": terms_list,
        "query_type": query_type,
        "is_exhaustive": is_exhaustive,
        "min_patent_years": min_patent_years,
        "limit": limit,
        "skip_cache": skip_cache,
        "term_field": term_field,
    }
