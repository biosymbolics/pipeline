"""
Patent graph reports
"""

from typing import Sequence
import logging
import networkx as nx

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from typings.patents import PatentApplication

from .types import Node, SerializableGraph

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def graph_patent_relationships(
    patents: Sequence[PatentApplication],
) -> SerializableGraph:
    """
    Graph UMLS ancestory for a set of patents

    - Nodes are UMLS entities
    - Edges are ancestory relationships OR patent co-occurrences (e.g. PD and anti-alpha-synucleins)
    - Nodes contain metadata of the relevant patents

    Process:
        - Step 1: get all UMLS entities associated with each patent
        - Step 2: get all UMLS ancestory relationships between those entities (ancestors from umls_lookup, other relationships from umls_graph)
        - Step 3: Create NetworkX graph from relationships
        - Step 4: Add co-occurrence edges to graph
        - Step 5: Add patent metadata and size attribute to nodes

    TODO: inefficient to load patents just to get ids
    """
    patent_ids = tuple([p["publication_number"] for p in patents])

    # TODO: save relationship type, cooccurrence, and ancestors
    sql = f"""
        select
            head_name as head,
            tail_name as tail,
            count(*) as weight,
            array_agg(publication_number) as patent_ids
        from annotations a, terms t, umls_graph g
        where a.publication_number = ANY({patent_ids})
        and a.term_id = t.id
        and g.head_id = ANY(t.ids)
        group by head_name, tail_name, rel_type
        order by count(*) desc
    """
    relationships = PsqlDatabaseClient().select(sql)
    G = nx.Graph()
    G.add_edges_from([(r["head"], r["tail"]) for r in relationships])
    # nx.set_node_attributes(G, bb, "betweenness") bb is dict

    link_data = nx.node_link_data(G)

    return SerializableGraph(
        links=link_data["links"],
        nodes=[
            Node(
                id=n["id"],
                patent_ids=n["patent_ids"],
                size=len(n["patent_ids"]),
            )
            for n in link_data["nodes"]
        ],
    )
