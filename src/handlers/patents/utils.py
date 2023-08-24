from handlers.patents.types import PatentSearchParams, ParsedPatentSearchParams


def parse_params(
    params: PatentSearchParams, default_max_results: int = 1000
) -> ParsedPatentSearchParams:
    terms = params.get("terms")
    terms_list = terms.split(";") if terms else []
    domains = params.get("domains") or None
    domains_list = domains.split(";") if domains else None

    min_patent_years = params.get("min_patent_years") or 10
    max_results = params.get("max_results") or default_max_results
    skip_cache = params.get("skip_cache") or False

    return {
        "terms": terms_list,
        "domains": domains_list,
        "min_patent_years": min_patent_years,
        "max_results": max_results,
        "skip_cache": skip_cache,
    }
