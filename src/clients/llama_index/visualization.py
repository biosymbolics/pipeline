# pylint: disable=E0102
"""
Visualizations
"""
from pyvis.network import Network
from llama_index.indices.knowledge_graph import GPTKnowledgeGraphIndex
from multipledispatch import dispatch

from .llama_index import load_index


@dispatch(object)  # type: ignore[no-redef]
def visualize_network(index: GPTKnowledgeGraphIndex):
    """
    Visualize network

    Args:
        index (GPTKnowledgeGraphIndex): kg index to visualize
    """
    graph = index.get_networkx_graph()
    net = Network(directed=True)
    net.from_nx(graph)
    net.show("graph.html", notebook=False)


@dispatch(str, str)  # type: ignore[no-redef]
def visualize_network(namespace: str, index_id: str):
    """
    Visualize network

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    index = load_index(namespace, index_id)
    if not index:
        raise Exception("index not found")
    # pylint: disable=E1120
    visualize_network(index)


# @dispatch(str)  # type: ignore[no-redef]
# def visualize_network(namespace: str):
#     """
#     Visualize network for composed indices within a namespace

#     Args:
#         namespace (str): namespace of the index (e.g. SEC-BMY)
#
#     """
#     composed = compose_graph(namespace)
#     if not composed:
#         raise Exception("composed graph not found")
#     visualize_network(composed)


def list_triples(namespace: str, index_id: str):
    """
    List triples
    """
    index = load_index(namespace, index_id)
    graph = index.get_networkx_graph()

    triples = []
    for edge in graph.edges(data=True):
        subject = edge[0]
        obj = edge[1]
        relationship = edge[2]["title"]
        triples.append((subject, relationship, obj))

    for triple in triples:
        print(triple)
