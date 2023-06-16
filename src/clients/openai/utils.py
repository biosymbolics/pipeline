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

    Example:
        >>> obj_str = __remove_comment_syntax('```json\n{"k01":"t1","k02":"t2"}``` ```json\n{"k11":"t1","k12":"t2"},{"k21":"t1","k22":"t2"}```')
        >>> json.loads(obj_str)
        {'k11': 't1', 'k12': 't2'}, {'k21': 't1', 'k22': 't2'}
    """
    json_blocks = re.findall(r"```json(.*?)```", text, re.DOTALL)
    if len(json_blocks) == 0:
        return text
    elif len(json_blocks) > 1:
        return json_blocks[-1]  # return the last

    return json_blocks[0]


def __load_json_array(text: str) -> list[str]:
    """
    (For typing)
    """
    array = json.loads(text)
    if not isinstance(array, list):
        raise Exception("Answer is not an array")
    return array


def __parse_answer_array(text: str, output_parser):
    # https://github.com/hwchase17/langchain/issues/1976
    logging.info("Naively parsing answer as array")
    try:
        parse_pipeline = [__remove_comment_syntax, __load_json_array]
        array = reduce(lambda x, f: f(x), parse_pipeline, text)  # type: ignore
        logging.info("ARRAY? %s", array)
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
        is_array (bool): whether to parse as array (special handling required)
        return_orig_on_fail (bool): whether to return original text on failure
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
