import json

from clients.patents.types import QueryType, get_query_type
from handlers.patents.types import PatentSearchParams, ParsedPatentSearchParams


def parse_bool(value: bool | str | None) -> bool:
    if value is None:
        return False
    return json.loads(str(value).lower())


def parse_params(
    params: PatentSearchParams, default_limit: int = 1000
) -> ParsedPatentSearchParams:
    terms = params.get("terms")
    terms_list = terms.split(";") if terms else []

    is_exhaustive = parse_bool(params.get("is_exhaustive", "false"))
    min_patent_years = params.get("min_patent_years") or 10
    limit = params.get("limit") or default_limit
    skip_cache = parse_bool(params.get("skip_cache", "false"))
    query_type: QueryType = get_query_type(params.get("query_type"))

    return {
        "terms": terms_list,
        "query_type": query_type,
        "is_exhaustive": is_exhaustive,
        "min_patent_years": min_patent_years,
        "limit": limit,
        "skip_cache": skip_cache,
    }
