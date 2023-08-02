"""
Functions specific to knowledge graph indices
"""
from llama_index.indices.knowledge_graph import GPTKnowledgeGraphIndex
import logging

from clients.llama_index.context import StorageArgs
from typings.indices import NamespaceKey
from prompts import BIOMEDICAL_TRIPLET_EXTRACT_PROMPT
from .llama_index_client import upsert_index, query_index

MAX_TRIPLES = 400


def create_and_query_kg_index(
    query: str,
    index_name: str,
    namespace_key: NamespaceKey,
    documents: list[str],
    storage_args: StorageArgs = {},
) -> str:
    """
    Creates or gets the kg index and queries

    Args:
        query (str): natural language query
        index_name (str): name of the index
        namespace_key (NamespaceKey): namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        documents (Document): list of llama_index Documents
        storage_args: storage args for loading index
    """
    index = create_kg_index(index_name, namespace_key, documents, storage_args)
    answer = query_index(index, query)
    logging.info("Answer: %s", answer)
    return answer


def create_kg_index(
    index_name: str,
    namespace_key: NamespaceKey,
    documents: list[str],
    storage_args: StorageArgs = {},
) -> GPTKnowledgeGraphIndex:
    """
    Creates the kg index if nx, and returns

    Args:
        index_name (str): name of the index
        namespace_key (NamespaceKey) namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        documents (Document): list of llama_index Documents
        storage_args: storage args for loading index
    """
    return upsert_index(
        index_name,
        documents,
        index_impl=GPTKnowledgeGraphIndex,  # type: ignore
        index_args={
            "kg_triple_extract_template": BIOMEDICAL_TRIPLET_EXTRACT_PROMPT,
            "max_knowledge_triplets": MAX_TRIPLES,
            "include_embeddings": True,
        },
        # model
        storage_args=storage_args,
    )
