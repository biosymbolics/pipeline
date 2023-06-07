"""
EntityIndex
"""
from typing import Optional
from llama_index import GPTVectorStoreIndex
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_REFINE_PROMPT_TMPL,
)
from langchain.output_parsers import ResponseSchema
import logging

from clients.llama_index import (
    get_output_parser,
    get_vector_index,
    parse_answer,
    query_index,
)
from common.utils.string import get_id
from sources.sec.prompts import GET_BIOMEDICAL_ENTITY_TEMPLATE
from types.indices import LlmIndex, NamespaceKey

from .types import is_entity_obj, EntityObj

ROOT_ENTITY_DIR = "entities"


class EntityIndex:
    """
    EntityIndex

    An index for a single entity, e.g. intervention BIBB122
    """

    def __init__(
        self, entity_name: str, source: NamespaceKey, canonical_id: Optional[str] = None
    ):
        """
        Initialize EntityIndex

        Args:
            entity_name (str): entity name
            source (NamespaceKey): source of the entity
            canonical_id (Optional[str], optional): canonical id of the entity. Defaults to None.
        """
        self.orig_entity_id = entity_name
        self.parsed_entity_id: Optional[str] = None  # this gets set later
        self.canonical_id = canonical_id
        self.type = "intervention"
        self.source = source

    @property
    def entity_id(self) -> str:
        """
        Returns entity id (parsed name if existing, otherwise original)
        """
        return self.parsed_entity_id or self.orig_entity_id

    def __maybe_set_parsed_id(self, new_parsed_id: str) -> None:
        """
        Set parsed entity name if not set
        """
        if not self.parsed_entity_id:
            self.parsed_entity_id = new_parsed_id

    def __get_response_schemas(self) -> list[ResponseSchema]:
        """
        Get response schemas for this entity
        """
        response_schemas = [
            ResponseSchema(name="name", description=f"normalized {self.type} name"),
            ResponseSchema(
                name="details", description=f"details about this {self.type}"
            ),
        ]
        return response_schemas

    def __get_entity_namespace(self) -> NamespaceKey:
        """
        Get namespace for the entity, e.g.

        For example, an entity record
            - for intervention BIBB122
            - based on an SEC 10-K

        would have namespace: ("entities", "intervention", "BIIB122", "BIBB", "SEC", "10-K")
        """
        return (ROOT_ENTITY_DIR, get_id(self.entity_id), self.type, *self.source)

    def __describe_entity_via_source(self, source_index: LlmIndex) -> EntityObj:
        """
        Get the description of an entity by querying the source index

        Args:
            source_index (LlmIndex): source index (e.g. an index for an SEC 10-K filing)
        """
        # get prompt to get details about entity
        query = GET_BIOMEDICAL_ENTITY_TEMPLATE(self.entity_id)

        # get the answer as json
        output_parser = get_output_parser(self.__get_response_schemas())
        fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
        fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
        qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
        refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)

        response = query_index(
            source_index, query, prompt=qa_prompt, refine_prompt=refine_prompt
        )

        logging.debug("Response from query_index: %s", response)

        # parse response into obj
        entity_obj = parse_answer(response, output_parser, return_orig_on_fail=False)

        # type check
        if not is_entity_obj(entity_obj):
            raise Exception(f"Failed to parse entity {self.entity_id}")
        return entity_obj

    def add_index(
        self,
        source_index: LlmIndex,
        index_id: str,
    ) -> Optional[GPTVectorStoreIndex]:
        """
        Add an index for this entity based on the source index,
        for example, an index for intervention BIBB122 based on some SEC 10-K filings
        """
        # get entity details by querying the source index
        entity_obj = self.__describe_entity_via_source(source_index)

        parsed_name = entity_obj["name"]
        details = entity_obj["details"]

        # maybe set parsed name
        self.__maybe_set_parsed_id(parsed_name)

        # btw uses self.parsed_entity_name (sketchy)
        entity_ns = self.__get_entity_namespace()

        # create index
        index = get_vector_index(entity_ns, index_id, [details])

        return index
