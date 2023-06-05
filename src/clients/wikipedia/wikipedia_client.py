from typing import Collection, Optional
from pydash import compact
import wptools as wikipedia


def fetch_infoboxes(category_name: str) -> list[dict]:
    """
    Fetch wikipedia infoboxes from category

    Args:
        category_name (str): Category name
    """

    def __fetch_pages(members: Collection) -> list[str]:
        titles = [m.title for m in members]
        return titles

    categories = fetch_category_members(category_name)
    titles = __fetch_pages(categories)
    infoboxes = compact([fetch_page_infobox(title) for title in titles])
    return infoboxes


def fetch_category_members(category_name: str) -> list[str]:
    """
    Fetch wikipedia category members

    Args:
        category_name (str): Category name
    """
    c_page = wikipedia.page(f"Category:{category_name}")
    c_page.get_more()

    if c_page.data:
        return c_page.data["categories"]
    return []


def fetch_page_infobox(title: str) -> Optional[dict]:
    """
    Fetch wikipedia infobox

    Args:
        title (str): Title
    """
    page = wikipedia.page(title)
    page.get_parse()

    if page.data:
        infobox = page.data["infobox"]
        print("%s - %s" % (title, infobox))
        return infobox

    return None
