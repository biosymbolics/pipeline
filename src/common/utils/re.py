def get_or_re(re_strs: list[str]) -> str:
    """
    ORs re_strs
    """
    return "(?:" + "|".join(re_strs) + ")"


WORD_CHAR_RE = "[\\w\u0370-\u03FF]"  # (includes greek chars)
WORD_DIGIT_CHAR_RE = "[\\d\\w\u0370-\u03FF]"
