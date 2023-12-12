import json
import logging

from typings.client import (
    OptionalRawPatentSearchParams as OptionalParams,
    PatentSearchParams,
    RawPatentSearchParams as RawParams,
)
from utils.args import parse_bool

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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

    logger.info("SKIP CACHE %s", parse_bool(p.skip_cache))
    return PatentSearchParams(
        **{
            "terms": terms_list,
            "exemplar_patents": exemplar_patents_list,
            "query_type": p.query_type,
            "min_patent_years": p.min_patent_years,
            "limit": limit,
            "skip_cache": parse_bool(p.skip_cache),
            "term_field": p.term_field,
        }
    )
