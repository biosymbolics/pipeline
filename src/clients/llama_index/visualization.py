"""
KG and other visualizations
"""
from typing import cast
from pyvis.network import Network
from llama_index.indices.knowledge_graph import GPTKnowledgeGraphIndex

from .persistence import load_index
from types.indices import NamespaceKey


def visualize_network_by_index(index: GPTKnowledgeGraphIndex):  # type: ignore
    """
    Visualize network

    Args:
        index (GPTKnowledgeGraphIndex): kg index to visualize
    """
    graph = index.get_networkx_graph()
    net = Network(directed=True)
    net.from_nx(graph)
    net.show("graph.html", notebook=False)


def visualize_network(namespace_key: NamespaceKey, index_id: str):
    """
    Visualize network

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    index = load_index(namespace_key, index_id)
    if not index:
        raise Exception("index not found")
    visualize_network_by_index(cast(GPTKnowledgeGraphIndex, index))


def list_triples_by_index(index: GPTKnowledgeGraphIndex):
    """
    List triples

    Args:
        index (GPTKnowledgeGraphIndex): kg index to visualize
    """
    graph = index.get_networkx_graph()

    triples = []
    for edge in graph.edges(data=True):
        subject = edge[0]
        obj = edge[1]
        relationship = edge[2]["title"]
        triples.append((subject, relationship, obj))

    for triple in triples:
        print(triple)


def list_triples(namespace_key: NamespaceKey, index_id: str):
    """
    List triples

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    index = load_index(namespace_key, index_id)
    kg_index = cast(GPTKnowledgeGraphIndex, index)
    list_triples_by_index(kg_index)
