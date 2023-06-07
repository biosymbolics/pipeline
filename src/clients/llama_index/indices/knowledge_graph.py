"""
Functions specific to knowledge graph indices
"""
from typing import Optional
from llama_index.indices.knowledge_graph import GPTKnowledgeGraphIndex
import logging

from clients.llama_index.constants import DEFAULT_MODEL_NAME
from clients.llama_index.types import LlmModel, NamespaceKey
from sources.sec.prompts import BIOMEDICAL_TRIPLET_EXTRACT_PROMPT
from .general import get_or_create_index, query_index

MAX_TRIPLES = 400


def create_and_query_kg_index(
    query: str,
    namespace_key: NamespaceKey,
    index_id: str,
    documents: list[str],
    model_name: Optional[LlmModel] = DEFAULT_MODEL_NAME,
) -> str:
    """
    Creates or gets the kg index and queries

    Args:
        query (str): natural language query
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
        model_name (LlmModel): model name
    """
    index = get_kg_index(namespace_key, index_id, documents, model_name=model_name)
    answer = query_index(index, query)
    logging.info("Answer: %s", answer)
    return answer


def get_kg_index(
    namespace_key: NamespaceKey,
    index_id: str,
    documents: list[str],
    model_name: Optional[LlmModel] = DEFAULT_MODEL_NAME,
) -> GPTKnowledgeGraphIndex:
    """
    Creates the kg index if nx, and returns

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
        model_name (LlmModel): model name
    """
    return get_or_create_index(
        namespace_key,
        index_id,
        documents,
        index_impl=GPTKnowledgeGraphIndex,  # type: ignore
        index_args={
            "kg_triple_extract_template": BIOMEDICAL_TRIPLET_EXTRACT_PROMPT,
            "max_knowledge_triplets": MAX_TRIPLES,
            "include_embeddings": True,
        },
        model_name=model_name,
    )
