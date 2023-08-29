"""
Client for querying SEC docs
"""
from datetime import date
from functools import partial
from typing import Any, TypedDict
from llama_index import Prompt
from llama_index.prompts.prompt_type import PromptType
from langchain.output_parsers import ResponseSchema
from clients.finance.yfinance_client import fetch_yfinance_data

from clients.llama_index.parsing import get_prompts_and_parser
from clients.low_level.boto3 import retrieve_with_cache_check
from core import SourceDocIndex
from core.indices.entity_index import EntityIndex
from utils.date import format_date, parse_date
from utils.misc import dict_to_named_tuple
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

    def __init__(self):
        self.source_index = SourceDocIndex(model_name="ChatGPT")
        self.entity_index = EntityIndex(model_name="ChatGPT")
        self.source = {"doc_source": "SEC", "doc_type": "10-K"}

    def ask_question(self, question: str, skip_cache: bool = False) -> str:
        """
        Ask a question about the source doc
        """
        key = get_id({**self.source, "question": question})
        ask = partial(
            self.source_index.query,
            question,
            source=dict_to_named_tuple(self.source),
            prompt_template=DEFAULT_TEXT_QA_PROMPT,
        )
        if skip_cache:
            return ask()

        return retrieve_with_cache_check(ask, key=key)

    def ask_about_entity(self, question: str) -> str:
        """
        Ask a question about the entity index
        """
        si_answer = self.entity_index.query(question, dict_to_named_tuple(self.source))
        return si_answer

    def get_events(
        self,
        ticker: str,
        start_date: date,
        end_date: date = date.today(),
        skip_cache: bool = False,
    ) -> StockPriceWithEvents:
        """
        Get SEC events atop stock perf for a given ticker symbol
        """
        key = get_id(
            {
                **self.source,
                "ticker": ticker,
                "start_date": start_date,
                "end_date": end_date,
            }
        )
        response_schemas = [
            ResponseSchema(name="date", description=f"event date (YYYY-MM-DD)"),
            ResponseSchema(name="details", description=f"details about this event"),
        ]
        prompts, parser = get_prompts_and_parser(response_schemas)
        question = f"""
            For the pharma company represented by the stock symbol {ticker},
            list important events such as regulatory approvals, trial readouts, acquisitions, reorgs, etc.
            that occurred between dates {format_date(start_date)} and {format_date(end_date)}.
            """

        def __get_event() -> StockPriceWithEvents:
            si_answer = self.source_index.query(
                question,
                source=dict_to_named_tuple(self.source),
                prompt_template=prompts[0],
            )

            events = parse_answer(si_answer, parser, is_array=True)
            stock_prices = fetch_yfinance_data(ticker, start_date, end_date)
            return {"stock": stock_prices, "events": events}

        if skip_cache:
            return __get_event()

        return retrieve_with_cache_check(__get_event, key=key)
