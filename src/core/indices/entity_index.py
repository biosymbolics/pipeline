"""
EntityIndex
"""
from datetime import datetime
from typing import Callable, Optional
from llama_index import GPTVectorStoreIndex
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_REFINE_PROMPT_TMPL,
)
from langchain.output_parsers import ResponseSchema
import logging
from pydash import flatten

from clients.llama_index import (
    get_index,
    query_index,
    get_output_parser,
    parse_answer,
    upsert_index,
)
from clients.llama_index.context import (
    DEFAULT_CONTEXT_ARGS,
    ContextArgs,
)
from clients.llama_index.types import DocMetadata
from clients.vector_dbs.pinecone import get_metadata_filters
from common.utils.misc import dict_to_named_tuple
from common.utils.namespace import get_namespace_id
from common.ner import extract_named_entities
from common.utils.string import get_id
from local_types.indices import NamespaceKey
from prompts import GET_BIOMEDICAL_ENTITY_TEMPLATE

from .source_doc_index import SourceDocIndex
from .types import is_entity_obj, EntityObj

INDEX_NAME = "entity-docs"
ENTITY_INDEX_CONTEXT_ARGS = DEFAULT_CONTEXT_ARGS


def create_entities_from_docs(
    section_map: dict[str, list[str]], get_namespace_key: Callable[[str], NamespaceKey]
):
    """
    Create entity index from a map of sections

    Args:
        section_map (dict[str, list[str]]): map of sections
        get_namespace_key (Callable[[str], NamespaceKey]): function to get namespace id from key, e.g.
            ``` python
                def get_namespace_key(key: str) -> NamespaceKey:
                    return dict_to_named_tuple(
                        {
                            "company": "PFE",
                            "doc_source": "SEC",
                            "doc_type": "10-K",
                            "period": key,
                        }
                    )

                create_from_docs(section_map, get_namespace_key)
            ```
    """
    all_sections = flatten(section_map.values())
    entities = extract_named_entities(all_sections)

    # this is the slow part
    for key, sections in section_map.items():
        ns_key = get_namespace_key(key)
        create_entity_indices(
            entities=entities,
            namespace_key=ns_key,
            documents=sections,
        )


def create_entity_indices(
    entities: list[str],
    namespace_key: NamespaceKey,
    documents: list[str],
):
    """
    For each entity in the provided list, summarize based on the document and persist in an index

    Args:
        entities (list[str]): list of entities to get indices for
        namespace_key (NamespaceKey) namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        documents (Document): list of llama_index Documents
    """
    index = SourceDocIndex()
    index.add_documents(namespace_key, documents)
    for entity in entities:
        try:
            idx = EntityIndex()
            idx.add_node(entity, index, namespace_key)
        except Exception as e:
            logging.error(f"Error creating entity index for {entity}: {e}")


class EntityIndex:
    """
    EntityIndex

    An index for a single entity across different docs/dates
    """

    def __init__(
        self,
        context_args: ContextArgs = ENTITY_INDEX_CONTEXT_ARGS,
    ):
        """
        Initialize EntityIndex

        Args:
            context_args (ContextArgs): context args. Defaults to ENTITY_INDEX_CONTEXT_ARGS.
        """
        self.context_args = context_args
        self.index = None
        self.index_impl = GPTVectorStoreIndex
        self.type = "intervention"

        self.__load()

    @property
    def __response_schemas(self) -> list[ResponseSchema]:
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

    def __get_namespace(
        self,
        source: NamespaceKey,
        entity_id: Optional[str] = None,
    ) -> NamespaceKey:
        """
        Namespace for the entity, e.g.

        For example, an entity record
            - for intervention BIBB122
            - based on an SEC 10-K

        would have namespace: ("entities", "intervention", "BIIB122", "BIBB", "SEC", "10-K")
        """
        entity_ns = (
            {
                "entity": get_id(entity_id),
                "entity_type": self.type,
            }
            if entity_id
            else {}
        )
        ns = {
            **source._asdict(),
            **entity_ns,
        }

        return dict_to_named_tuple(ns)

    def __describe_entity_by_source(
        self, entity_id: str, source_index: SourceDocIndex, source: NamespaceKey
    ) -> EntityObj:
        """
        Get the description of an entity by querying the source index

        Args:
            entity_id (str): entity id (e.g. BIBB122)
            source_index (LlmIndex): source index (e.g. an index for an SEC 10-K filing)
            source (NamespaceKey): namespace of the source
        """
        # get prompt to get details about entity
        query = GET_BIOMEDICAL_ENTITY_TEMPLATE(entity_id)

        # get the answer as json
        output_parser = get_output_parser(self.__response_schemas)
        fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
        fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
        qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
        refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)

        response = source_index.query(
            query, source, prompt=qa_prompt, refine_prompt=refine_prompt
        )

        logging.info("Response from query_index: %s", response)

        # parse response into obj
        entity_obj = parse_answer(response, output_parser, return_orig_on_fail=False)

        if not is_entity_obj(entity_obj):
            raise Exception(f"Failed to parse entity %s", entity_id)
        return entity_obj

    def __load(self):
        """
        Load entity index from disk
        """
        index = get_index(INDEX_NAME, **self.context_args.storage_args)
        self.index = index

    def add_node(
        self,
        entity_id: str,
        source_index: SourceDocIndex,
        source: NamespaceKey,
        retrieval_date: datetime = datetime.now(),
        canonical_id: str = "",
    ):
        """
        Create a node for this entity based on the source index,
        for example, an index for intervention BIBB122 based on some SEC 10-K filings

        Args:
            entity_id (str): entity id (e.g. BIBB122)
            source (NamespaceKey): namespace of the source
            source_index (LlmIndex): source index (e.g. an index for an SEC 10-K filing)
            retrieval_date (datetime, optional): retrieval date. Defaults to datetime.now()
            canonical_id (str, optional): canonical id. Defaults to "".
        """
        # get entity details by querying the source index
        entity_obj = self.__describe_entity_by_source(entity_id, source_index, source)
        name = entity_obj["name"]
        details = entity_obj["details"]

        # add metadata to the index (in particular, source which acts as namespace)
        def __get_metadata(doc) -> DocMetadata:
            return {
                **source._asdict(),
                entity_id: entity_id,
                "canonical_id": canonical_id,
                # parsed name; may differ by source and from entity_id
                "entity_name": name or "",
                # "retrieval_date": retrieval_date.isoformat(), # llamaindex gets mad
            }

        # uniq doc id for deduplication/idempotency
        def __get_doc_id(doc) -> str:
            return entity_id + "-" + get_namespace_id(source)

        if name is None or details is None:
            logging.warning(f"Skipping {entity_id} due to missing name or details")
            return

        upsert_index(
            INDEX_NAME,
            [details],
            index_impl=self.index_impl,  # type: ignore
            get_doc_metadata=__get_metadata,
            get_doc_id=__get_doc_id,
            context_args=self.context_args,
        )

    def add_node_from_docs(
        self,
        entity_id: str,
        source: NamespaceKey,
        documents: list[str],
    ):
        """
        Create a node for this entity based on the supplied documents

        Args:
            entity_id (str): entity id (e.g. BIBB122)
            source (NamespaceKey): source of the entity (named tuple; order and key names matter)
            documents (list[str]): list of documents
        """
        index = SourceDocIndex()
        index.add_documents(source, documents)
        self.add_node(entity_id, index, source)

    def query(
        self,
        query_string: str,
        source: NamespaceKey,
        entity_id: Optional[str] = None,
    ) -> str:
        """
        Query the entity index

        Args:
            query_string (str): query string
            source (NamespaceKey): source of the entity (named tuple; order and key names matter)
            entity_id (str): entity id (e.g. BIBB122). Optional; defaults to None.
        """
        metadata_filters = get_metadata_filters(self.__get_namespace(source, entity_id))

        if not self.index:
            raise ValueError("No index found.")

        answer = query_index(
            self.index, query_string, metadata_filters=metadata_filters
        )
        return answer
