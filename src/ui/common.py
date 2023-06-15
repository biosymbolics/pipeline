"""
Common UI functions
"""
from typing import Optional
import logging


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


def get_horizontal_list(items: list[str], label: Optional[str] = None) -> str:
    """
    Format a list of strings as a horizontal list (markdown representation)

    Args:
        items (list[str]): list of strings
        label (str, optional): label for list. Defaults to None.

    Returns:
        str: markdown representation of horizontal list
    """
    if len(items) == 0:
        logging.debug("No items to format as horizontal list")
        return ""

    md_list = " ".join([f"`{item}`" for item in items])

    if label:
        return f"**{label}**: " + md_list

    return md_list
