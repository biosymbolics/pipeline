"""
Core SEC methods
"""

from datetime import date, datetime

from common.utils.date import format_date_range
from sources.sec.sec_client import fetch_sec_docs


def fetch_quarterly_reports(
    ticker: str, start_date: date, end_date: date = datetime.now()
):
    """
    Get the R&D pipeline for a given company
    """
    date_range = format_date_range(start_date, end_date)
    critiera = [
        f"ticker:{ticker}",
        "filedAt:{" + date_range + "}",
        'formType:"10-Q"',
    ]
    quarterly_reports = fetch_sec_docs(critiera)
    return quarterly_reports
