"""
Core SEC methods
"""

from datetime import date, datetime
import traceback
from typing import Callable, Optional
import logging

from clients.sec import fetch_sec_docs, sec_client
from common.utils.date import format_date_range
from sources.sec.types import SecFiling


def __get_base_criteria(ticker: str, start_date: date, end_date: date) -> list[str]:
    date_range = format_date_range(start_date, end_date)
    criteria = [f"ticker:{ticker}", "filedAt:{" + date_range + "}"]
    return criteria


def fetch_quarterly_reports(
    ticker: str, start_date: date, end_date: date = datetime.now()
) -> list[SecFiling]:
    """
    Fetch quarterly SEC reports (10Q)
    """
    criteria = __get_base_criteria(ticker, start_date, end_date)
    quarterly_reports = fetch_sec_docs([*criteria, 'formType:"10-Q"'])
    return quarterly_reports


def fetch_annual_reports(
    ticker: str, start_date: date, end_date: date = datetime.now()
) -> list[SecFiling]:
    """
    Fetch annual SEC reports (10K)

    TODO: 20-F (foreign)
    """
    criteria = __get_base_criteria(ticker, start_date, end_date)
    reports = fetch_sec_docs([*criteria, 'formType:"10-K"'])

    return reports


def fetch_annual_reports_sections(
    ticker: str,
    start_date: date,
    end_date: date = datetime.now(),
    formatter: Optional[Callable] = None,
) -> dict[str, list[str]]:
    """
    Fetch annual SEC reports (10K) with sections

    Args:
        ticker (str): Ticker
        start_date (date): Start date
        end_date (date, optional): End date. Defaults to datetime.now().
        formatter (Optional[Callable], optional): Formatter. Defaults to None.

    Returns (dict[str, list[SecFiling]]): Map of sections to reports
    """
    reports = fetch_annual_reports(ticker, start_date, end_date)
    section_map = {}

    for report in reports:
        try:
            report_url = report.get("linkToHtml")
            sections = sec_client.extract_sections(
                report_url,
                return_type="html",
                formatter=formatter,
                sections=["1", "7"],
            )
            section_map[report.get("periodOfReport")] = sections
        except Exception as ex:
            logging.error("Error creating index for %s: %s", ticker, ex)
            traceback.print_exc()

    return section_map


def fetch_8k_reports(
    ticker: str, start_date: date, end_date: date = datetime.now()
) -> list[SecFiling]:
    """
    Fetch “current report” SEC docs (8-K)
    8-Ks are used to announce shareholder-relevant major events.
    """
    criteria = __get_base_criteria(ticker, start_date, end_date)
    reports = fetch_sec_docs([*criteria, 'formType:"8-K"'])

    return reports
