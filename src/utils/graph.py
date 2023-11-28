from itertools import islice
from multiprocessing import Pool
import networkx as nx
from networkx.classes.reportviews import NodeView
from random import sample


def chunk_nodes(nodes: list[NodeView], n: int):
    """
    Divide a list of nodes  in `n` chunks
    """
    l_c = iter(nodes)
    while 1:
        x = tuple(islice(l_c, n))
        if not x:
            return
        yield x


def betweenness_centrality_parallel(
    G: nx.Graph, k: int | None = None, num_process: int = 6
) -> dict[str, float]:
    """
    Parallelized betweenness centrality (normalized)

    Performance:
    - Naive impl cost: O(n^3)
    - Probable impl cost: O(nm+n^2log n). (n nodes, m edges), maybe less
    (https://cs-people.bu.edu/edori/betweenness.pdf - but we should actually check the NetworkX docs)

    Args:
        G: graph
        k: number of nodes to include in the map (for performance reasons)
        num_process: number of processes to use for parallelization
    """
    if k is not None:
        selected_nodes = sample(list(G.nodes()), k)
    else:
        selected_nodes = list(G.nodes())

    p = Pool(processes=num_process)
    node_divisor = len(p._pool) * 4  # type: ignore
    node_chunks = list(chunk_nodes(selected_nodes, len(selected_nodes) // node_divisor))
    num_chunks = len(node_chunks)
    bt_sc = p.starmap(
        nx.betweenness_centrality_subset,
        zip(
            [G] * num_chunks,
            node_chunks,
            [list(G)] * num_chunks,
            [True] * num_chunks,
            [None] * num_chunks,
        ),
    )

    # Reduce the partial solutions
    bt_c = dict(bt_sc[0])
    for bt in bt_sc[1:]:
        for n in bt:
            bt_c[n] += bt[n]

    return bt_c
