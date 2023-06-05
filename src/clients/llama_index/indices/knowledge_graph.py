"""
Functions specific to knowledge graph indices
"""
from llama_index.indices.knowledge_graph import GPTKnowledgeGraphIndex

from clients.llama_index.indices.vector import get_vector_index
from sources.sec.prompts import BIOMEDICAL_TRIPLET_EXTRACT_PROMPT
from .general import get_or_create_index, query_index

MAX_TRIPLES = 400


def create_and_query_kg_index(
    query: str, namespace: str, index_key: str, documents: list[str]
) -> str:
    """
    Creates the kg index if nx, and queries

    Args:
        query (str): natural language query
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
    """
    index = get_kg_index(namespace, index_key, documents)
    answer = query_index(index, query)
    return answer


def get_kg_index(
    namespace: str, index_id: str, documents: list[str]
) -> GPTKnowledgeGraphIndex:
    """
    Creates or returns the kg index if nx, and queries

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
    """
    return get_or_create_index(
        namespace,
        index_id,
        documents,
        index_impl=GPTKnowledgeGraphIndex,  # type: ignore
        index_args={
            "kg_triple_extract_template": BIOMEDICAL_TRIPLET_EXTRACT_PROMPT,
            "max_knowledge_triplets": MAX_TRIPLES,
            "include_embeddings": True,
        },
    )
