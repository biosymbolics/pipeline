def get_or_re(re_strs: list[str]) -> str:
    """
    ORs re_strs
    """
    return "(?:" + "|".join(re_strs) + ")"
