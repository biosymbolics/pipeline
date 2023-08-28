"""
Client for querying SEC docs
"""
from datetime import date
from typing import Any, TypedDict
from llama_index import Prompt
from llama_index.prompts.prompt_type import PromptType
from langchain.output_parsers import ResponseSchema
from clients.finance.yfinance_client import fetch_yfinance_data

from clients.llama_index.parsing import get_prompts_and_parser
from core import SourceDocIndex
from core.indices.entity_index import EntityIndex
from utils.date import format_date, parse_date
from utils.misc import dict_to_named_tuple
from utils.parse import parse_answer

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

    def ask_question(self, question: str) -> str:
        source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
        si_answer = self.source_index.query(
            question, source, prompt_template=DEFAULT_TEXT_QA_PROMPT
        )
        return si_answer

    def ask_about_entity(self, question: str) -> str:
        source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
        si_answer = self.entity_index.query(question, source)
        return si_answer

    def get_events(
        self, ticker: str, start_date: date, end_date: date = date.today()
    ) -> StockPriceWithEvents:
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
        source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
        si_answer = self.source_index.query(
            question,
            source,
            prompt_template=prompts[0],
        )

        events = parse_answer(si_answer, parser, is_array=True)  # type: ignore

        stock_prices = fetch_yfinance_data(ticker, start_date, end_date)
        return {"stock": stock_prices, "events": events}
