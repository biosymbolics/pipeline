"""
Utils for parsing llama index answers
"""
import logging
from typing import Any
from llama_index.output_parsers import LangchainOutputParser as OutputParser
from langchain.output_parsers import StructuredOutputParser, ResponseSchema


def get_output_parser(schemas: list[ResponseSchema]) -> OutputParser:
    """
    Get output parser for this entity

    Args:
        schemas (list[ResponseSchema]): response schemas
    """
    output_parser = StructuredOutputParser.from_response_schemas(schemas)
    output_parser = OutputParser(output_parser)
    return output_parser


def parse_answer(
    text: str, output_parser: OutputParser, return_orig_on_fail: bool = True
) -> Any:
    """
    Parses a text answer from llama index

    Args:
        text (str): text to parse
        output_parser (OutputParser): output parser to use (optional)

    TODO: handle typing
    """
    try:
        parsed = output_parser.parse(text)
        return parsed
    except Exception as ex:
        logging.error("Could not parse answer (%s): %s", text, ex)
        if not return_orig_on_fail:
            raise ex

    return text
