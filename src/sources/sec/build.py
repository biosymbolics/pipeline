"""
SEC build
"""
from datetime import date, datetime
import re

from common.utils.html_parsing.html import strip_inline_styles
from common.utils.misc import dict_to_named_tuple
from core.indices import entity_index
from typings.indices import NamespaceKey

from .sec import fetch_annual_reports_with_sections as fetch_annual_reports


def __format_for_ner(doc: str) -> str:
    """
    Format HTML content for NER processing
    - strip styles
    - remove certain characters

    Args:
        doc (str): HTML content
    """
    stop_words = ["^", "ª", "*", "--"]
    escaped_stop_words = [re.escape(word) for word in stop_words]
    stop_patterns = ["\\([0-9a-z]{1}\\)"]
    pattern = re.compile("|".join([*escaped_stop_words, *stop_patterns]))

    stripped = strip_inline_styles(doc)
    cleaned = pattern.sub("", stripped)
    return cleaned


def build_indices(ticker: str, start_date: date, end_date: date = datetime.now()):
    """
    Create knowledge graph from documents

    Args:
        ticker: stock ticker for company (e.g. BMY, PFE)
        start_date (date): start date
        end_date (date, optional): end date, defaults to now
    """
    section_map = fetch_annual_reports(
        ticker, start_date, end_date, formatter=__format_for_ner
    )

    def get_namespace_key(key: str) -> NamespaceKey:
        return dict_to_named_tuple(
            {
                "company": ticker,
                "doc_source": "SEC",
                "doc_type": "10-K",
                "period": key,
            }
        )

    entity_index.create_from_docs(section_map, get_namespace_key)
