"""
SourceDocIndex
"""
from datetime import datetime
import logging
from typing import Any, Optional, Type
from llama_index import VectorStoreIndex

from clients.llama_index import (
    load_index,
    query_index,
    upsert_index,
    NerKeywordTableIndex,
)
from clients.llama_index.context import StorageArgs
from clients.llama_index.types import DocMetadata
from clients.stores import pinecone
from common.utils.namespace import get_namespace_id
from constants.core import DEFAULT_MODEL_NAME
from typings.indices import LlmIndex, LlmModelType, NamespaceKey, Prompt, RefinePrompt

INDEX_NAME = "source-docs"

ENTITY_TYPES = ["compounds", "diseases", "mechanisms"]


DEFAULT_STORAGE_ARGS: StorageArgs = {
    "storage_type": "mongodb",
    "ner_options": {
        "use_llm": False,
    },
}


class SourceDocIndex:
    """
    SourceDocIndex

    Simple index over raw-ish source docs
    """

    def __init__(
        self,
        storage_args: StorageArgs = DEFAULT_STORAGE_ARGS,
        model_name: LlmModelType = DEFAULT_MODEL_NAME,
        index_impl: Type[LlmIndex] = NerKeywordTableIndex,
    ):
        """
        initialize index

        Args:
            storage_args (StorageArgs, optional): storage args. Defaults to DEFAULT_STORAGE_ARGS.
            model_name (str, optional): model name. Defaults to DEFAULT_MODEL_NAME.
            index_impl (LlmIndex, optional): index implementation. Defaults to NerKeywordTableIndex.
        """
        self.storage_args = storage_args
        self.model: LlmModelType = model_name
        self.index_impl = index_impl
        self.index = None
        self.__load()

    def __load(self):
        """
        Load source doc index from disk
        """
        index = load_index(
            INDEX_NAME,
            self.index_impl,
            self.model,
            self.storage_args,
            index_args={"entity_types": ENTITY_TYPES},
        )
        self.index = index

    def __get_metadata_filters(self, source: NamespaceKey):
        """
        Get metadata filters for source

        TODO: only works for VectorStoreIndex currently!!
        """
        if isinstance(self.index_impl, VectorStoreIndex):
            return pinecone.get_metadata_filters(source)

        return source._asdict()

    def add_documents(
        self,
        source: NamespaceKey,
        documents: list[str],
        retrieval_date: datetime = datetime.now(),
    ):
        """
        Load docs into index with metadata

        Args:
            documents (list[str]): list of documents
            source (NamespaceKey): source namespace
            retrieval_date (datetime, optional): retrieval date of source docs. Defaults to datetime.now().
        """

        def __get_metadata(doc) -> DocMetadata:
            return {
                **source._asdict(),
                "retrieval_date": retrieval_date.isoformat(),  # TODO: this can mess up pinecone
            }

        # uniq doc id for deduplication/idempotency
        def __get_doc_id(doc) -> str:
            return get_namespace_id(source)

        index = upsert_index(
            INDEX_NAME,
            documents,
            index_impl=self.index_impl,
            get_doc_metadata=__get_metadata,
            get_doc_id=__get_doc_id,
            model_name=self.model,
            storage_args=self.storage_args,
            index_args={"entity_types": ENTITY_TYPES},
        )
        self.index = index

    def query(
        self,
        query_string: str,
        source: NamespaceKey,
        prompt_template: Optional[Prompt] = None,
        refine_prompt: Optional[RefinePrompt] = None,
    ) -> Any:
        """
        Query the index

        Args:
            query_string (str): query string
            source (NamespaceKey): source namespace that acts as filter, e.g.
                ``` python
                {
                    "company": "BIBB",
                    "doc_source": "SEC",
                    "doc_type": "10-K",
                    "period": "2020-12-31",
                }
                ```
            prompt_template (Prompt, optional): prompt. Defaults to None.
            refine_prompt (RefinePrompt, optional): refine prompt. Defaults to None.
        """
        if not self.index:
            raise ValueError("Index not initialized.")

        metadata_filters = self.__get_metadata_filters(source)

        if not isinstance(metadata_filters, dict):
            logging.info(
                "Querying with filters (if vector store) %s",
                metadata_filters.__dict__.items(),
            )

        answer = query_index(
            self.index,
            query_string,
            prompt_template=prompt_template,
            refine_prompt=refine_prompt,
            metadata_filters=metadata_filters,
        )

        return answer
