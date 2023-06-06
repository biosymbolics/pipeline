"""
SEC build
"""
from datetime import date, datetime
import re
from pydash import flatten
import logging

from clients.llama_index.indices.entity import get_entity_indices
from common.ner import extract_named_entities
from common.utils.html_parsing.html import strip_inline_styles

from .sec import fetch_annual_reports_with_sections as fetch_annual_reports


def __format_for_ner(doc: str) -> str:
    """
    Format HTML content for NER processing
    - strip styles
    - remove certain characters
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
        end_date (date): end date
    """
    section_map = fetch_annual_reports(
        ticker, start_date, end_date, formatter=__format_for_ner
    )

    all_sections = flatten(section_map.values())
    entities = extract_named_entities(all_sections, "spacy")
    logging.info("ENTITIES: %s", entities)

    # this is the slow part
    for period, sections in section_map.items():
        get_entity_indices(
            entities=entities,
            namespace=ticker,
            index_id=period,
            documents=sections,
        )
