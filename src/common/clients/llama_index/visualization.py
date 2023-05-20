from pyvis.network import Network
from llama_index.indices.knowledge_graph import GPTKnowledgeGraphIndex
from multipledispatch import dispatch

from .llama_index import load_index

@dispatch(object)
def visualize_network(index: GPTKnowledgeGraphIndex):
    """
    Visualize network

    Args:
        index (GPTKnowledgeGraphIndex): kg index to visualize
    """
    g = index.get_networkx_graph()
    net = Network(directed=True)
    net.from_nx(g)
    net.show("graph.html", notebook=False)


@dispatch(str, str)
def visualize_network(namespace: str, index_id: str):
    """
    Visualize network

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    index = load_index(namespace, index_id)
    visualize_network(index)