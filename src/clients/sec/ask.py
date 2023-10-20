"""
Client for querying SEC docs
"""
from datetime import date
from typing import Any, TypedDict
from llama_index import Prompt
from llama_index.prompts.prompt_type import PromptType
from langchain.output_parsers import ResponseSchema
from clients.finance.yfinance_client import fetch_yfinance_data
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

from clients.low_level.boto3 import retrieve_with_cache_check
from utils.date import format_date
from utils.parse import parse_answer
from utils.string import get_id

StockPriceWithEvents = TypedDict("StockPriceWithEvents", {"stock": Any, "events": Any})

DEFAULT_TEXT_QA_PROMPT_TMPL = """
    Context information is below.
    ---------------------
    {context_str}
    ---------------------
    Given the context information and not prior knowledge,
    provided a detailed, scientific and accurate answer to the question below.
    Format the answer in markdown, and include tables, lists and links where appropriate.
    Tag entities (compounds, drugs and diseases) with [E].
    ---------------------
    {query_str}
    ---------------------
"""
DEFAULT_TEXT_QA_PROMPT = Prompt(
    DEFAULT_TEXT_QA_PROMPT_TMPL, prompt_type=PromptType.QUESTION_ANSWER
)


class AskSecClient:
    """
    Client for querying SEC docs
    """

    def get_events(
        self,
        ticker: str,
        start_date: date,
        end_date: date = date.today(),  # TODO
        skip_cache: bool = False,
    ) -> StockPriceWithEvents:
        """
        Get SEC events atop stock perf for a given ticker symbol
        """
        key = get_id(
            {
                "company": ticker,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        response_schemas = [
            ResponseSchema(name="date", description=f"event date (YYYY-MM-DD)"),
            ResponseSchema(name="details", description=f"details about this event"),
        ]
        parser = StructuredOutputParser.from_response_schemas(response_schemas)
        question = f"""
            For the pharma company represented by the stock symbol {ticker},
            list important events such as regulatory approvals, trial readouts, acquisitions, reorgs, etc.
            that occurred between dates {format_date(start_date)} and {format_date(end_date)}.
            """
        prompt = ""

        def __get_event() -> StockPriceWithEvents:
            si_answer = ""  # GPTClient

            events = parse_answer(si_answer, parser, is_array=True)  # type: ignore
            stock_prices = fetch_yfinance_data(ticker, start_date, end_date)
            return {"stock": stock_prices, "events": events}

        if skip_cache:
            return __get_event()

        return retrieve_with_cache_check(__get_event, key=key)
