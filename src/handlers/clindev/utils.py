from typings.client import RawTrialSearchParams, TrialSearchParams
from utils.args import parse_bool


def parse_params(
    _params: RawTrialSearchParams,
    default_limit: int = 800,
) -> TrialSearchParams:
    """
    Parse patent params
    """
    # combine default and provided params
    p = RawTrialSearchParams(**_params.__dict__)

    # parse ";"-delimited terms
    terms_list = [t.strip() for t in (p.terms.split(";") if p.terms else [])]

    limit = p.limit or default_limit

    return TrialSearchParams(
        **{
            "terms": terms_list,
            "query_type": p.query_type,
            "limit": limit,
            "skip_cache": parse_bool(p.skip_cache),
        }
    )
