"""
SEC build
"""
from datetime import date, datetime
import logging

from clients.llama_index import get_keyword_index
from clients.sec import sec_client
from common.utils.html_parsing.html import strip_inline_styles
from sources.sec.rd_pipeline import fetch_annual_reports


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
                report_url, return_type="html", formatter=strip_inline_styles
            )
            index = get_keyword_index(
                namespace=ticker,
                index_id=report.get("periodOfReport"),
                documents=sections,
            )
        except Exception as ex:
            logging.error("Error creating index for %s: %s", ticker, ex)
