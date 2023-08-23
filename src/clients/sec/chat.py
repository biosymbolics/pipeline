"""
Client for querying SEC docs
"""
from datetime import date
from llama_index import Prompt
from llama_index.prompts.prompt_type import PromptType

from core import SourceDocIndex
from utils.misc import dict_to_named_tuple
from utils.date import format_date


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


class SecChatClient:
    """
    Client for querying SEC docs
    """

    def __init__(self):
        self.source_index = SourceDocIndex(model_name="GPT4")

    def ask_question(self, question: str) -> str:
        source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
        si_answer = self.source_index.query(
            question, source, prompt_template=DEFAULT_TEXT_QA_PROMPT
        )
        return si_answer

    def get_events(
        self, ticker: str, start_date: date, end_date: date = date.today()
    ) -> str:
        prompt = (
            f"""
            For the pharma company represented by the stock symbol {ticker},
            list important events such as regulatory approvals, trial readouts, acquisitions, reorgs, etc.
            that occurred between dates {format_date(start_date)} and {format_date(end_date)}
            as json in the form """
            + '{ "YYYY-MM-DD": "the event" }.'
        )
        source = dict_to_named_tuple({"doc_source": "SEC", "doc_type": "10-K"})
        si_answer = self.source_index.query(prompt, source)
        return si_answer
