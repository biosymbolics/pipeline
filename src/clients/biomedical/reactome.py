"""
Reactome client
"""
from collections.abc import MutableMapping
import json
from typing import Optional
from typing_extensions import TypedDict
import logging
import regex as re
import requests
from pydash import flatten

from utils.list import dedup

Node = TypedDict("Node", {"id": str, "parent": Optional[str], "name": str})
ReactomeNode = TypedDict("ReactomeNode", {"displayName": str, "stId": str})
Lineage = list[ReactomeNode]
TreeMap = MutableMapping[str, Node]
TreeRecords = list[Node]

REACTOME_ID_PREFIX = "R-HSA"
REACTOME_ID_RE = "[A-Z]-[A-Z]{3}-[0-9]{6}"

REACTOME_URL = "https://reactome.org/ContentService/data/event/"


def __get_url(reactome_id: str) -> str:
    """
    Get url for entity

    Args:
        reactome_id (str): reactome id
    """
    return f"{REACTOME_URL}/{reactome_id}/ancestors"


def __extract_reactome_id(entity_id: str) -> str:
    """
    Extract reactome id from entity id or term
    e.g. REACTOME:R-HSA-418822 -> R-HSA-418822
        NTN1:DCC oligomer:p-Y397-PTK2 [plasma membrane] R-HSA-418822 -> R-HSA-418822

    Args:
        entity_id (str): Entity id or term (e.g. REACTOME:R-HSA-418822 or GO:123456)
    """
    match = re.findall(REACTOME_ID_RE, entity_id)
    return match[0] if match else entity_id


def __extract_reactome_ids(entity_ids: list[str]) -> list[str]:
    """
    Extract reactome ids from a list of entity ids (not all are ids) and dedups.

    Example:
    ```
    ["REACTOME:R-HSA-418822", "GO:123456", "REACTOME:R-HSA-418822"] -> ["R-HSA-418822", "GO:123456"]
    ```

    Args:
        entity_ids (list[str]): List of entity ids
    """
    reactome_ids = [
        __extract_reactome_id(id) for id in entity_ids if REACTOME_ID_PREFIX in id
    ]
    return dedup(reactome_ids)


def __get_lineage(reactome_id: str) -> Lineage:
    """
    Call Reactome to get hierarchy for a given entity

    Args:
        reactome_id (str): Reactome id
    """
    url = __get_url(reactome_id)
    response = requests.get(url)
    response.raise_for_status()

    lineage = flatten(json.loads(response.text))
    lineage.reverse()

    return lineage


def __add_lineage(lineage: Lineage, tree: TreeMap) -> TreeMap:
    """
    Add a lineage into the tree.

    Order defines parent/child relationship (e.g. [A, B, C] -> A is parent of B, B is parent of C)

    Args:
        lineage (Lineage): Lineage
        tree (TreeMap): Tree
    """
    previous_node = None

    for node in lineage:
        node_id = node["stId"]
        if node_id not in tree:
            tree[node_id] = {
                "id": node_id,
                "name": node["displayName"],
                "parent": None,
            }

        if previous_node:
            tree[node_id] = {
                **tree[node_id],
                "parent": previous_node["displayName"],
            }

        previous_node = node

    return tree


def __get_lineages(reactome_ids: list[str]) -> list[Lineage]:
    """
    Get reactome hierachies by id

    Args:
        reactome_ids (list[str]): List of reactome ids
    """
    return [__get_lineage(id) for id in reactome_ids]


def __create_tree(lineages: list[Lineage]) -> TreeRecords:
    """
    Create a tree out of multiple lineages

    Args:
        lineages (list[Lineage]): List of lineages
    """
    tree: TreeMap = {}
    for lineage in lineages:
        tree = __add_lineage(lineage, tree)

    return list(tree.values())


def fetch_reactome_tree(entity_ids: list[str]) -> TreeRecords:
    """
    Fetch reactome tree for a list of entity ids

    Args:
        entity_ids (list[str]): List of entity ids (which may contain reactome ids)
    """
    reactome_ids = __extract_reactome_ids(entity_ids)
    entities = __get_lineages(reactome_ids)
    tree = __create_tree(entities)

    return tree
