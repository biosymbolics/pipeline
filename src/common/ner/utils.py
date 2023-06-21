"""
Utils for the NER pipeline
"""
from common.utils.re import (
    get_or_re,
    ReCount,
    ALPHA_CHARS,
)

"""
Re utils
"""
# end-of-entity regex
EOE_RE = "\\b" + ".*"

# start-of-entity regex
SOE_RE = ".*"


def get_entity_re(
    core_entity_re: str,
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    is_case_insensitive=False,
) -> str:
    """
    Returns a regex for an entity with a start-of-entity (soe_re) and end-of-entity (eoe_re) regexes

    Args:
        core_entity_re (str): regex for the core entity
        soe_re (str): start-of-entity regex (default: SOE_RE)
        eoe_re (str): end-of-entity regex (default: EOE_RE)
        is_case_insensitive (bool): whether to make the regex case insensitive (default: False)
    """
    core_re = soe_re + core_entity_re + eoe_re

    if is_case_insensitive:
        return "(?i)" + core_re

    return core_re


def get_infix_entity_re(
    core_infix_res: list[str],
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    count: ReCount = 2,
) -> str:
    """
    Returns a regex for an entity with a prefix and suffix

    Args:
        core_infix_res (list[str]): list of regexes for infixes
        soe_re (str): start-of-entity regex (default: SOE_RE)
        eoe_re (str): end-of-entity regex (default: EOE_RE)
        count (ReCount): number of alpha chars in prefix and suffix
    """
    return (
        soe_re
        + ALPHA_CHARS(count)
        + get_or_re(core_infix_res)
        + ALPHA_CHARS(count)
        + eoe_re
    )


def get_suffix_entitiy_re(
    core_suffix_res: list[str],
    soe_re: str = SOE_RE,
    eoe_re: str = EOE_RE,
    prefix_count: ReCount = 2,
) -> str:
    """
    Returns a regex for an entity with a prefix and suffix

    Args:
        core_suffix_res (list[str]): list of regexes for suffixes
        soe_re (str): start-of-entity regex (default: SOE_RE)
        eoe_re (str): end-of-entity regex (default: EOE_RE)
        prefix_count (ReCount): number of alpha chars in prefix
    """
    return soe_re + ALPHA_CHARS(prefix_count) + get_or_re(core_suffix_res) + eoe_re
