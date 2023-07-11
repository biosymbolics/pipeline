"""
HTML extraction utilities
"""
import logging
from bs4 import BeautifulSoup

DEFAULT_REMOVE_TAGS: list[str] = [
    "head",
    "script",
    "style",
    "hr",
    "br",
    "noscript",
    "template",
]
MAX_RECURSION_DEPTH = 100


class ContentExtractor:
    """
    Content extractor for HTML
    Attempts a reasonable extraction of content from tables
    """

    def __init__(self, remove_tags=None):
        """
        Initialize content extractor
        """
        self.remove_tags = remove_tags or DEFAULT_REMOVE_TAGS

    def __join(self, texts, separator=" "):
        """
        Join a list of strings with a separator
        """
        return separator.join(text for text in texts if text)

    def __table_to_text(self, node):
        """
        Convert a table to text
        """
        rows = node.find_all("tr")
        return self.__join([self.__row_to_text(row) for row in rows], ".\n ")

    def __row_to_text(self, node):
        """
        Convert a row to text
        """
        cells = node.find_all("td")
        return self.__join(
            [self.__text_node_to_text(cell, ", ") for cell in cells], ", "
        )

    def __text_node_to_text(self, node, separator=" "):
        """
        Convert a text node to text
        """
        node_text = node.get_text(strip=False, separator=separator)
        node_text = node_text.strip()
        return node_text

    def __element_to_text(self, node, depth=0):
        """
        Convert an element to text
        """
        if depth > MAX_RECURSION_DEPTH:
            logging.warning("Exceeded max recursion depth, returning empty string")
            return ""

        if node.name == "table":
            text = self.__table_to_text(node)
        elif node.name == "tr":
            text = self.__row_to_text(node)
        elif node.name in ["div", "body", "html", "section", "iframe"]:
            children = [
                child
                for child in node.children
                if child.name and child not in (node, node.parent)
            ]
            if not children:
                text = self.__text_node_to_text(node)
            else:
                text = self.__join(
                    [self.__element_to_text(child, depth + 1) for child in children]
                )
        else:
            text = self.__text_node_to_text(node)

        return text

    def __call__(self, string):
        """
        Extract content from HTML string
        """
        texts = self.parse_html(string)
        return texts

    def parse_html(self, html_string):
        """
        Parse HTML string into list of text strings

        - remove "remove_tags"
        - split into top-level nodes
        - convert each node to text
        """
        soup = BeautifulSoup(html_string, "html.parser")

        for removed_tag in self.remove_tags:
            for element in soup.find_all(removed_tag):
                element.decompose()

        top_level_nodes = soup.find_all(recursive=False)
        html_texts = [self.__element_to_text(node) for node in top_level_nodes]

        return html_texts


def extract_text(html: str) -> str:
    """
    Strip HTML tags from html, returning content

    Args:
        html (str): the html
    """
    extractor = ContentExtractor()
    return " ".join(extractor(html))


def strip_inline_styles(html: str) -> str:
    """
    Strip inline styles from html

    Args:
        html (str): the html
    """
    soup = BeautifulSoup(html, "html.parser")

    # Create a new BeautifulSoup object with the desired changes
    [element.attrs.pop("style", None) for element in soup.select("[style]")]

    # Return the modified HTML
    return str(soup)
