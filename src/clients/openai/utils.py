"""
Utils for OpenAI client
"""

from typing import Any
from langchain.output_parsers import StructuredOutputParser
import logging


def parse_answer(
    text: str, output_parser: StructuredOutputParser, return_orig_on_fail: bool = True
) -> Any:
    """
    Parses a text answer from llama index

    Args:
        text (str): text to parse
        output_parser (OutputParser): output parser to use (optional)
    """
    try:
        parsed = output_parser.parse(text)
        return parsed
    except Exception as ex:
        logging.error("Could not parse answer (%s): %s", text, ex)
        if not return_orig_on_fail:
            raise ex

    return text
