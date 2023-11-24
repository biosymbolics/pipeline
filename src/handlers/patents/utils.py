import json

from handlers.patents.types import (
    OptionalRawPatentSearchParams as OptionalParams,
    PatentSearchParams,
    RawPatentSearchParams as RawParams,
)


def parse_bool(value: bool | str | None) -> bool:
    if value is None:
        return False
    return json.loads(str(value).lower())


def parse_params(
    _params: RawParams,
    default_params: OptionalParams = OptionalParams(),
    default_limit: int = 800,
) -> PatentSearchParams:
    """
    Parse patent params
    """
    # combine default and provided params
    p = RawParams(**{**default_params.__dict__, **_params.__dict__})

    # parse ";"-delimited terms
    terms_list = [t.strip() for t in (p.terms.split(";") if p.terms else [])]

    # exemplar patents
    exemplar_patents_list = [
        t.strip() for t in (p.exemplar_patents.split(";") if p.exemplar_patents else [])
    ]

    limit = p.limit or default_limit

    return PatentSearchParams(
        **{
            "terms": terms_list,
            "exemplar_patents": exemplar_patents_list,
            "query_type": p.query_type,
            "min_patent_years": p.min_patent_years,
            "limit": limit,
            "skip_cache": p.skip_cache,
            "term_field": p.term_field,
        }
    )
