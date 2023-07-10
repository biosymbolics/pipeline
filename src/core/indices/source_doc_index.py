"""
SourceDocIndex
"""
from datetime import datetime
import logging
from typing import Any, Optional, Type
from llama_index import GPTVectorStoreIndex

from clients.llama_index import load_index, query_index, upsert_index
from clients.llama_index.context import ContextArgs, DEFAULT_CONTEXT_ARGS
from clients.llama_index.types import DocMetadata
from clients.vector_dbs.pinecone import get_metadata_filters
from common.utils.namespace import get_namespace_id
from typings.indices import LlmIndex, NamespaceKey, Prompt, RefinePrompt

INDEX_NAME = "source-docs"


class SourceDocIndex:
    """
    SourceDocIndex

    Simple index over raw-ish source docs

    Pinecone query example:
    ``` json
    { "$and": [{ "company": "PFE" }, { "doc_source": "SEC" }, { "doc_type": "10-K" }, { "period": "2020-12-31" }] }
    ```

    """

    def __init__(
        self,
        context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
        index_impl: Type[LlmIndex] = GPTVectorStoreIndex,
    ):
        """
        initialize

        Args:
            context_args (ContextArgs, optional): context args. Defaults to DEFAULT_CONTEXT_ARGS.
            index_impl (LlmIndex, optional): index implementation. Defaults to GPTVectorStoreIndex.
        """
        self.context_args = context_args
        self.index = None
        self.index_impl = index_impl

        self.__load()

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
            source (NamespaceKey): source namespace, e.g.
                ``` python
                dict_to_named_tuple(
                    {
                        "company": "PFE",
                        "doc_source": "SEC",
                        "doc_type": "10-K",
                        "period": "2020-12-31",
                    }
                )
                ```
            retrieval_date (datetime, optional): retrieval date of source docs. Defaults to datetime.now().
        """

        def __get_metadata(doc) -> DocMetadata:
            return {
                **source._asdict(),
                # "retrieval_date": retrieval_date.isoformat(),
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
            context_args=self.context_args,
        )
        self.index = index

    def __load(self):
        """
        Load source doc index from disk
        """
        index = load_index(INDEX_NAME, **self.context_args.storage_args)
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
                dict_to_named_tuple(
                    {
                        "company": "BIBB",
                        "doc_source": "SEC",
                        "doc_type": "10-K",
                        "period": "2020-12-31",
                    }
                )
                ```
            prompt_template (Prompt, optional): prompt. Defaults to None.
            refine_prompt (RefinePrompt, optional): refine prompt. Defaults to None.
        """
        if not self.index:
            raise ValueError("Index not initialized.")

        metadata_filters = get_metadata_filters(source)

        logging.info("Querying with filters %s", metadata_filters.__dict__.items())

        answer = query_index(
            self.index,
            query_string,
            prompt_template=prompt_template,
            refine_prompt=refine_prompt,
            metadata_filters=metadata_filters,
        )

        return answer
