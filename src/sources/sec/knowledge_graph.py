from datetime import date, datetime

from common.clients.llama_index import get_or_create_index
from sources.sec.rd_pipeline import fetch_annual_reports

from . import sec_client


async def build_knowledge_graph(
    ticker: str, start_date: date, end_date: date = datetime.now()
):
    """
    Create knowledge graph from documents

    Args:
        ticker: stock ticker for company (e.g. BMY, PFE)
        start_date (date): start date
        end_date (date): end date
    """
    reports = fetch_annual_reports(ticker, start_date, end_date)

    for report in reports:
        report_url = report.get("linkToHtml")
        sec_section = sec_client.extract_section(report_url, return_type="text")
        get_or_create_index(
            namespace=ticker,
            index_id=report.get("periodOfReport"),
            documents=[sec_section],
        )
