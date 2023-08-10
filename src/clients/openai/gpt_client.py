"""
Client stub for GPT
"""
import os
from langchain import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import Any, Literal, Optional
import logging

from .utils import parse_answer

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3

GptModel = Literal["gpt-3.5-turbo", "gpt-4"]
DEFAULT_GPT_MODEL: GptModel = "gpt-3.5-turbo"


class GptApiClient:
    """
    Class for OpenAI API client
    """

    def __init__(
        self,
        schemas: Optional[list[ResponseSchema]] = None,
        model: GptModel = DEFAULT_GPT_MODEL,
    ):
        self.client = None
        self.model = model

        if schemas:
            prompt_template, output_parser = self.__get_schema_things(schemas)
            self.output_parser: Optional[StructuredOutputParser] = output_parser
        else:
            self.output_parser = None
            prompt_template = PromptTemplate(
                template="Answer this query as best as possible.\n{query}",
                input_variables=["query"],
            )

        self.prompt_template: PromptTemplate = prompt_template

    def __get_schema_things(
        self, schemas: list[ResponseSchema]
    ) -> tuple[PromptTemplate, StructuredOutputParser]:
        """
        Get prompt template and output parser from response schemas, respectively serving to:
          - modify the prompt to request the desired output format
          - parse the output to the desired format.
        """
        output_parser = StructuredOutputParser.from_response_schemas(schemas)
        format_intructions = output_parser.get_format_instructions()

        prompt_template = PromptTemplate(
            template="Answer this query as best as possible.\n{format_instructions}\n{query}",
            input_variables=["query"],
            partial_variables={"format_instructions": format_intructions},
        )
        return (prompt_template, output_parser)

    def __format_answer(self, answer: str, is_array: bool = False) -> Any:
        if self.output_parser:
            logging.debug("Formatting answer: %s", answer)
            return parse_answer(answer, self.output_parser, is_array, True)

        return answer

    def query(self, query: str, is_array: bool = False) -> Any:
        """
        Query GPT, applying the prompt template and output parser if response schemas were provided
        """
        input = self.prompt_template.format_prompt(query=query)
        chat_model = ChatOpenAI(
            temperature=0,
            client="chat",
            model=self.model,
            max_tokens=DEFAULT_MAX_TOKENS,
            openai_api_key=OPENAI_API_KEY,
        )
        output = chat_model(input.to_messages())
        try:
            return self.__format_answer(output.content, is_array=is_array)
        except Exception as e:
            logging.warning("Error formatting answer: %s", e)
            return output.content

    def describe_terms(
        self, terms: list[str], context_terms: Optional[list[str]] = None
    ) -> str:
        """
        Simple query to describe terms

        Args:
            terms (list[str]): list of terms to describe
            context_terms (list[str], optional): list of terms to provide context for the query
        """
        context_query = (
            " in the context of: " + ", ".join(context_terms) if context_terms else ""
        )
        multiple_query = " and how they relate" if len(terms) > 1 else ""
        query = f"""
            Provide succinct, technical information about the following{context_query}{multiple_query}:
            {", ".join(terms)}
        """
        return self.query(query)

    def describe_topic(self, topic_features: list[str]) -> str:
        """
        Simple query to describe a topic
        """
        query = (
            "Return a good, succinct name for the topic described by the following words:\n"
            + "\n".join(topic_features)
        )
        return self.query(query)
