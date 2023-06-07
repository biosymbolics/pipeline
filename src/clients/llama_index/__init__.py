"""
LlamaIndex client module
"""
from .visualization import visualize_network

from .parsing import get_output_parser, parse_answer
from .indices.composed import get_composed_index, query_composed_index
from .indices.general import create_index, get_index, query_index, get_or_create_index
from .indices.knowledge_graph import create_and_query_kg_index, get_kg_index
from .indices.vector import get_vector_index, create_and_query_vector_index

__all__ = [
    "create_index",
    "create_and_query_kg_index",
    "create_and_query_vector_index",
    "get_composed_index",
    "get_index",
    "get_or_create_index",
    "get_kg_index",
    "get_output_parser",
    "get_vector_index",
    "parse_answer",
    "query_index",
    "query_composed_index",
    "visualize_network",
]
