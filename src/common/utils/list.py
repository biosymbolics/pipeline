"""
Utils for lists/arrays
"""


def diff_lists(list_one: list, list_two: list) -> list:
    """
    Returns the items present in list_one but missing in list_two
    """
    set_two = set(list_two)
    dropped = [x for x in list_one if x not in set_two]
    return dropped


def dedup(a_list: list) -> list:
    """
    Returns a list with duplicates removed

    Args:
        a_list (list): list to deduplicate
    """
    return list(set(a_list))
