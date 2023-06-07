"""
EntityIndex
"""
from datetime import datetime
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
    create_index,
    get_output_parser,
    parse_answer,
)
from clients.llama_index.context import get_storage_context
from common.utils.string import get_id
from sources.sec.prompts import GET_BIOMEDICAL_ENTITY_TEMPLATE
from types.indices import LlmIndex, NamespaceKey

from .source_doc_index import SourceDocIndex
from .types import is_entity_obj, EntityObj

ROOT_ENTITY_DIR = "entities"


def create_entity_indices(
    entities: list[str],
    namespace_key: NamespaceKey,
    index_id: str,
    documents: list[str],
):
    """
    For each entity in the provided list, summarize based on the document and persist in an index

    Args:
        entities (list[str]): list of entities to get indices for
        namespace_key (NamespaceKey) namespace of the index (e.g. ("BIBB", "SEC", "10-K"))
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
    """
    index = SourceDocIndex(namespace_key, index_id, documents=documents)
    for entity in entities:
        idx = EntityIndex(entity, namespace_key)
        idx.add_node(index, index_id)


class EntityIndex:
    """
    EntityIndex

    An index for a single entity, e.g. intervention BIBB122
    """

    def __init__(
        self,
        entity_name: str,
        source: NamespaceKey,
        canonical_id: Optional[str] = None,
        retrieval_date: datetime = datetime.now(),
    ):
        """
        Initialize EntityIndex

        Args:
            entity_name (str): entity name
            source (NamespaceKey): source of the entity
            canonical_id (Optional[str], optional): canonical id of the entity. Defaults to None.
            retrieval_date (datetime, optional): retrieval date of the entity. Defaults to datetime.now().
        """
        self.orig_entity_id = entity_name
        self.parsed_entity_id: Optional[str] = None  # this gets set later
        self.ids: list[str] = []  # all entity ids
        self.canonical_id = canonical_id
        self.type = "intervention"
        self.source = source
        self.retrieval_date = retrieval_date

    @property
    def entity_id(self) -> str:
        """
        Returns entity id (parsed name if existing, otherwise original)
        """
        return self.parsed_entity_id or self.orig_entity_id

    def __add_id(self, new_id: str):
        """
        Add new entity id to the list

        Args:
            new_id (str): new entity id
        """
        ids = [*self.ids, new_id]
        self.ids = ids

    def __maybe_set_parsed_id(self, new_parsed_id: str) -> None:
        """
        Set parsed entity name if not set
        """
        if not self.parsed_entity_id:
            self.parsed_entity_id = new_parsed_id
        self.__add_id(new_parsed_id)

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

    def __describe_entity_by_source(self, source_index: SourceDocIndex) -> EntityObj:
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

        response = source_index.query(
            query, prompt=qa_prompt, refine_prompt=refine_prompt
        )

        logging.debug("Response from query_index: %s", response)

        # parse response into obj
        entity_obj = parse_answer(response, output_parser, return_orig_on_fail=False)

        # type check
        if not is_entity_obj(entity_obj):
            raise Exception(f"Failed to parse entity {self.entity_id}")
        return entity_obj

    def add_node(
        self,
        source_index: SourceDocIndex,
        index_id: str,
    ) -> Optional[GPTVectorStoreIndex]:
        """
        Create a node for this entity based on the source index,
        for example, an index for intervention BIBB122 based on some SEC 10-K filings

        Args:
            source_index (LlmIndex): source index (e.g. an index for an SEC 10-K filing)
            index_id (str): unique id of the index (e.g. 2020-01-1)
        """
        # get entity details by querying the source index
        entity_obj = self.__describe_entity_by_source(source_index)
        name, details = dict(entity_obj)

        # maybe set parsed name
        self.__maybe_set_parsed_id(name)

        entity_ns = self.__get_entity_namespace()

        # create index
        index = create_index(
            entity_ns,
            index_id,
            [details],
            index_impl=GPTVectorStoreIndex,  # type: ignore
            index_args={
                "storage_context": get_storage_context(entity_ns, store_type="pinecone")
            },
        )

        return index

    def add_node_from_docs(
        self,
        documents: list[str],
        index_id: str,
    ) -> Optional[GPTVectorStoreIndex]:
        """
        Create a node for this entity based on the supplied documents

        Args:
            documents (list[str]): list of documents
            index_id (str): unique id of the index (e.g. 2020-01-1)
        """
        index = SourceDocIndex(self.source, index_id, documents=documents)

        return self.add_node(index, index_id)
