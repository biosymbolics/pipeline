"""
LlamaIndex client module
"""
from .llama_index import (
    get_query_engine,
    query_index,
)
from .visualization import visualize_network

from .index_specific.knowledge_graph import create_and_query_kg_index, get_kg_index
from .index_specific.vector import get_vector_index, create_and_query_vector_index

__all__ = [
    "create_and_query_kg_index",
    "create_and_query_vector_index",
    "get_kg_index",
    "get_vector_index",
    "get_query_engine",
    "query_index",
    "visualize_network",
]
