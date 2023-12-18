from typings.client import EntitySearchParams, RawEntitySearchParams
from utils.args import parse_bool


def parse_params(
    _params: RawEntitySearchParams,
) -> EntitySearchParams:
    """
    Parse patent params
    """
    # combine default and provided params
    p = RawEntitySearchParams(**_params.__dict__)

    # parse ";"-delimited terms
    terms_list = [t.strip() for t in (p.terms.split(";") if p.terms else [])]

    return EntitySearchParams(
        **{
            "terms": terms_list,
            "query_type": p.query_type,
            "limit": p.limit,
            "skip_cache": parse_bool(p.skip_cache),
        }
    )
