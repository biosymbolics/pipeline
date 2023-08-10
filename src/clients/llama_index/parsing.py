"""
Utils for parsing llama index answers
"""
import logging
from typing import Any, NamedTuple
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from llama_index.output_parsers import LangchainOutputParser
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_REFINE_PROMPT_TMPL,
)

from utils.string import remove_comment_syntax


def get_output_parser(schemas: list[ResponseSchema]) -> LangchainOutputParser:
    """
    Get output parser for this entity

    Args:
        schemas (list[ResponseSchema]): response schemas
    """
    output_parser = StructuredOutputParser.from_response_schemas(schemas)
    output_parser = LangchainOutputParser(output_parser)
    return output_parser


PromptsAndParser = NamedTuple(
    "PromptsAndParser",
    [
        ("prompts", tuple[QuestionAnswerPrompt, RefinePrompt]),
        ("parser", LangchainOutputParser),
    ],
)


def get_prompts_and_parser(schemas: list[ResponseSchema]) -> PromptsAndParser:
    """
    Get prompts and parser for this entity, given response schemas

    Args:
        schemas (list[ResponseSchema]): response schemas

    Returns:
        prompts (tuple[QuestionAnswerPrompt, RefinePrompt]): prompts
    """
    output_parser = get_output_parser(schemas)
    fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
    fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
    qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
    refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)
    return PromptsAndParser((qa_prompt, refine_prompt), output_parser)


def parse_answer(
    text: str, output_parser: LangchainOutputParser, return_orig_on_fail: bool = True
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
