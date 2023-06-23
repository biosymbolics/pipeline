"""
Utils for OpenAI client
"""

from functools import reduce
from typing import Any
from langchain.output_parsers import StructuredOutputParser
import json
import re
import logging

from common.utils.string import remove_comment_syntax


def load_json_array(text: str) -> list[str]:
    """
    (For typing)
    """
    array = json.loads(text)
    if not isinstance(array, list):
        raise Exception("Answer is not an array")
    return array


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
        is_array (bool): whether to parse as array (special handling required)
        return_orig_on_fail (bool): whether to return original text on failure
    """

    def __parse_answer_array(text: str, output_parser):
        # https://github.com/hwchase17/langchain/issues/1976
        logging.info("Naively parsing answer as array")
        try:
            parse_pipeline = [remove_comment_syntax, load_json_array]
            array = reduce(lambda x, f: f(x), parse_pipeline, text)  # type: ignore
            final: list[dict] = [
                output_parser.parse("```json" + json.dumps(item) + "```")
                for item in array
            ]
            return final
        except json.JSONDecodeError:
            raise Exception("Answer is not valid json")

    try:
        if is_array:
            return __parse_answer_array(text, output_parser)

        return output_parser.parse(text)
    except Exception as ex:
        logging.error("Could not parse answer (%s): %s", text, ex)
        if not return_orig_on_fail:
            raise ex

    return text
