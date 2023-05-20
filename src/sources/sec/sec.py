"""
Core SEC methods
"""

from datetime import date, datetime

from common.utils.date import format_date_range
from sources.sec.sec_client import fetch_sec_docs
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
    """
    criteria = __get_base_criteria(ticker, start_date, end_date)
    annual_reports = fetch_sec_docs([*criteria, 'formType:"10-K"'])
    return annual_reports