"""
Common UI functions
"""
from typing import Optional
import logging
import urllib.parse


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


def get_horizontal_list(
    items: list[str], label: Optional[str] = None, url_base: Optional[str] = None
) -> str:
    """
    Format a list of strings as a horizontal list (markdown representation)

    Args:
        items (list[str]): list of strings
        label (str, optional): label for list. Defaults to None.
        url_base (str, optional): url base for each item. Defaults to None.

    Returns:
        str: markdown representation of horizontal list
    """
    if len(items) == 0:
        logging.debug("No items to format as horizontal list")
        return ""

    if url_base:
        items = [
            get_markdown_link(url_base + urllib.parse.quote(item), item)
            for item in items
        ]
        md_list = " ".join(items)
    else:
        md_list = " ".join([f"`{item}`" for item in items])

    if label:
        return f"**{label}**: " + md_list

    return md_list
