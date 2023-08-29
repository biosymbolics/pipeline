"""
Utils for parsing llama index answers
"""
import logging
from typing import NamedTuple, cast
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from llama_index.output_parsers import LangchainOutputParser
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_REFINE_PROMPT_TMPL,
)

from typings.gpt import OutputParser


def get_output_parser(schemas: list[ResponseSchema]) -> OutputParser:
    """
    Get output parser for this entity

    Args:
        schemas (list[ResponseSchema]): response schemas
    """
    output_parser = StructuredOutputParser.from_response_schemas(schemas)
    output_parser = LangchainOutputParser(output_parser)
    return cast(OutputParser, output_parser)


PromptsAndParser = NamedTuple(
    "PromptsAndParser",
    [
        ("prompts", tuple[QuestionAnswerPrompt, RefinePrompt]),
        ("parser", OutputParser),
    ],
)


def get_prompts_and_parser(
    schemas: list[ResponseSchema], qa_template=DEFAULT_TEXT_QA_PROMPT_TMPL
) -> PromptsAndParser:
    """
    Get prompts and parser for this entity, given response schemas

    Args:
        schemas (list[ResponseSchema]): response schemas

    Returns:
        prompts (tuple[QuestionAnswerPrompt, RefinePrompt]): prompts
    """
    output_parser = get_output_parser(schemas)
    fmt_qa_tmpl = output_parser.format(qa_template)
    fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
    qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
    refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)
    return PromptsAndParser((qa_prompt, refine_prompt), output_parser)
