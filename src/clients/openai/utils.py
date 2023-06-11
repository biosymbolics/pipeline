"""
Utils for OpenAI client
"""

from functools import reduce
from typing import Any
from langchain.output_parsers import StructuredOutputParser
import json
import re
import logging


def __remove_comment_syntax(text: str) -> str:
    """
    Remove leading ```json and trailing ``` (and anything after it)
    """
    cleaned = re.sub("```json|```.*", "", text, flags=re.DOTALL)
    return cleaned


def __load_json(text: str) -> str:
    return json.loads(text)


def __parse_answer_array(text: str, output_parser):
    # https://github.com/hwchase17/langchain/issues/1976
    logging.info("Naively parsing answer as array")
    try:
        parse_pipeline = [__remove_comment_syntax, __load_json]
        array = reduce(lambda x, f: f(x), parse_pipeline, text)
        final: list[dict] = [
            output_parser.parse("```json" + json.dumps(item) + "```") for item in array
        ]
        return final
    except json.JSONDecodeError:
        raise Exception("Answer is not valid json")


def parse_answer(
    text: str,
    output_parser: StructuredOutputParser,
    is_array: bool = False,
    return_orig_on_fail: bool = True,
) -> Any:
    """
    Parses a text answer from llama index

    Args:
        text (str): text to parse
        output_parser (OutputParser): output parser to use (optional)
    """
    try:
        if is_array:
            return __parse_answer_array(text, output_parser)

        return output_parser.parse(text)
    except Exception as ex:
        logging.error("Could not parse answer (%s): %s", text, ex)
        if not return_orig_on_fail:
            raise ex

    return text
