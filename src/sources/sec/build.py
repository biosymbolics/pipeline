"""
SEC build
"""
from datetime import date, datetime
import logging
import re

from clients.llama_index.indices.entity import get_entity_indices
from clients.sec import sec_client
from common.ner import extract_named_entities
from common.utils.file import save_json_as_file
from common.utils.html_parsing.html import strip_inline_styles
from sources.sec.rd_pipeline import fetch_annual_reports


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
    reports = fetch_annual_reports(ticker, start_date, end_date)

    for report in reports:
        try:
            report_url = report.get("linkToHtml")
            sections = sec_client.extract_sections(
                report_url,
                return_type="html",
                formatter=__format_for_ner,
                sections=["1", "7"],
            )
            # save_as_file("\n".join(sections), "sections.txt")
            entities = extract_named_entities(sections, "spacy")
            save_json_as_file(entities, f"{ticker}_{report.get('periodOfReport')}.json")

            get_entity_indices(
                entities=entities,
                namespace=ticker,
                index_id=report.get("periodOfReport"),
                documents=sections,
            )
        except Exception as ex:
            logging.error("Error creating index for %s: %s", ticker, ex)
            raise ex
