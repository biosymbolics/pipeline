"""
LlamaIndex client module
"""
from .visualization import visualize_network

from .indices.composed import get_composed_index, query_composed_index
from .indices.general import get_index, query_index, get_or_create_index
from .indices.keyword_table import create_and_query_keyword_index, get_keyword_index
from .indices.knowledge_graph import create_and_query_kg_index, get_kg_index
from .indices.vector import get_vector_index, create_and_query_vector_index

__all__ = [
    "create_and_query_keyword_index",
    "create_and_query_kg_index",
    "create_and_query_vector_index",
    "get_composed_index",
    "get_index",
    "get_or_create_index",
    "get_kg_index",
    "get_keyword_index",
    "get_vector_index",
    "query_index",
    "query_composed_index",
    "visualize_network",
]
