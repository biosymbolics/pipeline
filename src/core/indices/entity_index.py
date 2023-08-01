"""
EntityIndex
"""
from datetime import datetime
from typing import Callable, Optional, cast
from llama_index import GPTVectorStoreIndex, VectorStoreIndex
from langchain.output_parsers import ResponseSchema
from pydash import flatten
import logging

from clients.llama_index import (
    load_index,
    query_index,
    parse_answer,
    upsert_index,
)
from clients.llama_index.context import StorageArgs
from clients.llama_index.parsing import get_prompts_and_parser
from clients.llama_index.types import DocMetadata
from clients.stores.pinecone import get_metadata_filters
from common.ner.ner import NerTagger
from common.utils.misc import dict_to_named_tuple
from common.utils.namespace import get_namespace_id
from common.utils.string import get_id
from constants.core import DEFAULT_MODEL_NAME
from typings.indices import LlmModelType, NamespaceKey
from prompts import GET_BIOMEDICAL_ENTITY_TEMPLATE

from .source_doc_index import SourceDocIndex
from .types import is_entity_obj, EntityObj

INDEX_NAME = "entity-docs"


class EntityIndex:
    """
    EntityIndex

    An index for a single entity across different docs/dates
    """

    def __init__(
        self,
        model_name: LlmModelType = DEFAULT_MODEL_NAME,
        storage_args: StorageArgs = {},
    ):
        """
        Initialize EntityIndex

        Args:
            model_name (LlmModelType, optional): model name. Defaults to DEFAULT_MODEL_NAME.
            storage_args (StorageArgs, optional): storage args. Defaults to {}.
        """
        self.storage_args = storage_args
        self.model: LlmModelType = model_name
        self.index = None
        self.index_impl = GPTVectorStoreIndex
        self.__load()

    def __get_namespace(
        self,
        source: NamespaceKey,
        entity_id: Optional[str] = None,
        entity_type: Optional[str] = "intervention",
    ) -> NamespaceKey:
        """
        Namespace for the entity, e.g.

        For example, an entity record
            - for intervention BIBB122
            - based on an SEC 10-K

        would have namespace: ("entities", "intervention", "BIIB122", "BIBB", "SEC", "10-K")
        """
        ns = (
            {
                **source._asdict(),
                "entity": get_id(entity_id),
                "entity_type": entity_type,
            }
            if entity_id
            else source._asdict()
        )
        return dict_to_named_tuple(ns)

    def __get_response_schemas(self, entity_type: str) -> list[ResponseSchema]:
        """
        Get response schemas for this entity
        """
        response_schemas = [
            ResponseSchema(name="name", description=f"normalized {entity_type} name"),
            ResponseSchema(
                name="details", description=f"details about this {entity_type}"
            ),
        ]
        return response_schemas

    def __describe_entity_by_source(
        self,
        entity_id: str,
        source_index: SourceDocIndex,
        source: NamespaceKey,
        entity_type: str = "intervention",
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
        prompts, parser = get_prompts_and_parser(
            self.__get_response_schemas(entity_type)
        )
        response = source_index.query(query, source, *prompts)

        logging.info("Response from query_index: %s", response)

        # parse response into obj
        entity_obj = parse_answer(response, parser, return_orig_on_fail=False)

        if not is_entity_obj(entity_obj):
            raise Exception(f"Failed to parse entity %s", entity_id)
        return entity_obj

    def __load(self):
        """
        Load entity index from disk
        """
        index = load_index(INDEX_NAME, VectorStoreIndex, self.model, self.storage_args)
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
            index_impl=self.index_impl,
            get_doc_metadata=__get_metadata,
            get_doc_id=__get_doc_id,
            model_name=self.model,
            storage_args=self.storage_args,
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
        if not self.index:
            raise ValueError("No index found.")

        # filtering on namespace for precise and efficient retrieval
        metadata_filters = get_metadata_filters(self.__get_namespace(source, entity_id))

        answer = query_index(
            self.index, query_string, metadata_filters=metadata_filters
        )
        return answer

    @staticmethod
    def create_entity_indices(
        entities: list[str],
        namespace_key: NamespaceKey,
        documents: list[str],
    ):
        """
        For each entity in the provided list, summarize based on the document and persist in an index

        Args:
            entities (list[str]): list of entities
            namespace_key (NamespaceKey): namespace key
            documents (list[str]): list of documents
        """
        index = SourceDocIndex()
        index.add_documents(namespace_key, documents)

        for entity in entities:
            try:
                idx = EntityIndex()
                idx.add_node(entity, index, namespace_key)
            except Exception as e:
                logging.error(f"Error creating entity index for {entity}: {e}")

    @staticmethod
    def create_from_docs(
        doc_map: dict[str, list[str]], get_namespace_key: Callable[[str], NamespaceKey]
    ):
        """
        Create entity index from a map of docs

        Args:
            doc_map (dict[str, list[str]]): map of docs
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

                    create_from_docs(doc_map, get_namespace_key)
                ```
        """
        tagger = NerTagger.get_instance()
        for key, docs in doc_map.items():
            entities = tagger.extract(
                docs,
                entity_types=["mechanisms", "compounds", "classes"],
            )
            flattend_entities = [ent[0] for ent in flatten(entities)]
            ns_key = get_namespace_key(key)
            EntityIndex.create_entity_indices(
                entities=cast(list[str], flattend_entities),
                namespace_key=ns_key,
                documents=docs,
            )
