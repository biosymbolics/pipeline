import json

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

    domains = params.get("domains") or None
    domains_list = domains.split(";") if domains else None

    is_exhaustive = parse_bool(params.get("is_exhaustive", "false"))
    min_patent_years = params.get("min_patent_years") or 10
    limit = params.get("limit") or default_limit
    skip_cache = parse_bool(params.get("skip_cache", "false"))

    return {
        "terms": terms_list,
        "domains": domains_list,
        "is_exhaustive": is_exhaustive,
        "min_patent_years": min_patent_years,
        "limit": limit,
        "skip_cache": skip_cache,
    }
