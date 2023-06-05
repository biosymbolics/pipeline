"""
String utilities
"""


def get_id(string: str) -> str:
    """
    Returns the id of a string

    Args:
        string (str): string to get id of
    """
    return string.replace(" ", "_").lower()
