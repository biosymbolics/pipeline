"""
Common UI functions
"""
from typing import Optional


def get_markdown_link(url: str, text: Optional[str] = None):
    """
    Get markdown link

    Args:
        url (str): url
        text (str, optional): link text. Defaults to None.

    Returns:
        str: markdown link
    """
    return f"[{text or url}]({url})"
