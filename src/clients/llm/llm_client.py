"""
Client stub for GPT
"""

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.llms import HuggingFaceEndpoint
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from typing import Any, Literal, Optional, cast
import logging
from clients.low_level.boto3 import retrieve_with_cache_check

from typings.gpt import OutputParser
from utils.parse import parse_answer
from utils.string import get_id

DEFAULT_MAX_TOKENS = 1024
DEFAULT_TEMPERATURE = 0.3

# mixtral via Groq... cheap & fast.
# Mistral 8x7B is MoE
GptModel = Literal[
    "gpt-3.5-turbo",
    "gpt-4",
    "mixtral-8x7b-32768",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "biomistral-7b-dare-mps",
]
DEFAULT_GPT_MODEL: GptModel = "gpt-4"

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class GptApiClient:
    """
    Class for OpenAI API client
    """

    def __init__(
        self,
        schemas: Optional[list[ResponseSchema]] = None,
        model: GptModel = DEFAULT_GPT_MODEL,
        skip_cache: bool = False,
    ):
        self.model = model
        prompt_template, output_parser = self._get_prompt_and_parser(schemas)
        self.output_parser: Optional[StructuredOutputParser] = output_parser
        self.prompt_template: PromptTemplate = prompt_template
        self.skip_cache = skip_cache

    def _get_prompt_and_parser(
        self, schemas: list[ResponseSchema] | None
    ) -> tuple[PromptTemplate, StructuredOutputParser | None]:
        """
        Get prompt template and output parser from response schemas, respectively serving to:
          - modify the prompt to request the desired output format
          - parse the output to the desired format.
        """
        if not schemas:
            prompt_template = PromptTemplate(
                template="Answer this query to the best of your expert ability.\n{query}",
                input_variables=["query"],
            )
            return (prompt_template, None)

        output_parser = StructuredOutputParser.from_response_schemas(schemas)
        format_intructions = output_parser.get_format_instructions()

        prompt_template = PromptTemplate(
            template="Answer this query to the best of your expert ability.\n{format_instructions}\n{query}",
            input_variables=["query"],
            partial_variables={"format_instructions": format_intructions},
        )
        return (prompt_template, output_parser)

    def _format_answer(self, answer: str, is_array: bool = False) -> Any:
        if self.output_parser:
            logger.debug("Formatting answer: %s", answer)
            return parse_answer(
                answer, cast(OutputParser, self.output_parser), is_array, True
            )

        return answer

    def get_llm(self) -> BaseChatModel:
        if self.model == "mixtral-8x7b-32768":
            return ChatGroq(
                temperature=DEFAULT_TEMPERATURE, name=self.model, max_tokens=20000
            )

        if self.model == "mistralai/Mistral-7B-Instruct-v0.1":
            return HuggingFaceEndpoint(
                repo_id=self.model,
                temperature=DEFAULT_TEMPERATURE,
                max_new_tokens=8000,
                model_kwargs={"max_tokens": 20000},
            )

        if self.model == "biomistral-7b-dare-mps":
            return HuggingFaceEndpoint(
                endpoint_url="https://v0sk5eajj1ohvc6z.us-east-1.aws.endpoints.huggingface.cloud",
                temperature=DEFAULT_TEMPERATURE,
                max_new_tokens=8000,
                max_tokens=20000,
            )

        return ChatOpenAI(
            temperature=DEFAULT_TEMPERATURE,
            model=self.model,
            max_tokens=DEFAULT_MAX_TOKENS,
        )

    async def _query(self, query: str, is_array: bool = False) -> Any:
        """
        Query GPT, applying the prompt template and output parser if response schemas were provided
        """
        llm = self.get_llm()
        llm_chain = LLMChain(prompt=self.prompt_template, llm=llm)
        output = llm_chain.invoke(input={"query": query})
        try:
            return self._format_answer(output["text"], is_array=is_array)
        except Exception as e:
            logger.warning("Error formatting answer: %s", e)
            return output

    async def query(self, query: str, is_array: bool = False) -> Any:
        """
        Query GPT with a cache check
        """
        if self.skip_cache:
            result = self._query(query, is_array)
        else:
            key = f"llm-query-{get_id([query])}"
            result = await retrieve_with_cache_check(
                lambda: self._query(query, is_array), key=key
            )

        return result

    async def describe_terms(
        self,
        terms: list[str],
        context_terms: Optional[list[str]] = None,
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
        multiple_query = " , focusing on how they relate" if len(terms) > 1 else ""
        query = f"""
            Provide 1-2 paragraphs of scientific and technical information, avoiding self-reference,
            about the following{context_query}{multiple_query},
            in markdown:
            {", ".join(terms)}
        """
        return await self.query(query)

    async def describe_topic(self, topic_features: list[str]) -> str:
        """
        Simple query to describe a topic
        """
        query = (
            "Return a good, succinct name for the topic described by the following words:\n"
            + "\n".join(topic_features)
        )
        return await self.query(query)

    async def generate_ip_description(self, short_description: str) -> str:
        """
        Generate a description of IP based on a short sentence
        (for testing with company_finder)
        """
        query = (
            "Please expand the following into a 2-3 paragraph technical description of a biomedical invention:\n"
            + short_description
        )
        return await self.query(query)

    @staticmethod
    async def clindev_timelines(indication: str) -> list[dict]:
        """
        Query GPT about clindev timelines

        (TODO: move to a separate client)
        """
        prompt = (
            f"What is the typical clinical development timeline for indication {indication}? "
            "Return the answer as an array of json objects with the following fields: stage, offset, median_duration, iqr. "
        )

        response_schemas = [
            ResponseSchema(name="stage", description="e.g. Phase 1", type="string"),
            ResponseSchema(
                name="offset",
                description="equal to cumulative median duration of previous stages, 0 for the first stage.",
                type="number",
            ),
            ResponseSchema(
                name="median_duration",
                description="median duration of this stage in years (e.g. 2.5)",
                type="number",
            ),
            ResponseSchema(
                name="iqr",
                description="interquartile range of this stage's duration in years (e.g. 0.8)",
                type="number",
            ),
        ]

        gpt_client = GptApiClient(schemas=response_schemas)
        answer_as_array: list[dict] = await gpt_client.query(prompt, is_array=True)
        return answer_as_array
