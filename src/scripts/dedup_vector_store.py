"""
Dedup vector store
"""
import json
import sys

from pydash import flatten
from clients.vector_dbs.pinecone import get_vector_store
from core import SOURCE_DOC_INDEX_NAME
import logging
from collections import defaultdict

from system import initialize

PERIODS = [
    "2015-12-31",
    "2016-12-31",
    "2017-12-31",
    "2018-12-31",
    "2019-12-31",
    "2020-12-31",
    "2021-12-31",
    "2022-12-31",
]
TICKERS = ["pfe"]

NULL_VECTOR: list[float] = [0 for _ in range(1536)]


def __find_duplicates(nodes: list[dict]):
    def __get_key(node):
        ni = json.loads(node.metadata.get("node_info", {}))
        n_info = {**ni, "doc_id": node.metadata.get("ref_doc_id")}
        return tuple(n_info.items())

    def __get_dup_ids(key, nodes) -> list[str]:
        if len(nodes) > 1:
            to_save = nodes.pop(0)
            logging.info("Saving id %s", to_save["id"])
            return [node["id"] for node in nodes]
        return []

    node_groups = defaultdict(list)
    for node in nodes:
        key = __get_key(node)
        node_groups[key].append(node)

    duplicate_ids = flatten(
        [__get_dup_ids(key, node) for key, node in node_groups.items()]
    )

    return duplicate_ids


def dedup_documents(tickers: list[str] = TICKERS, periods: list[str] = PERIODS):
    """
    Deduplicate documents in vector store

    filter={ "ref_doc_id": { "$eq": "company-pfe-doc_source-sec-doc_type-10-k-period-2022-12-31" }}
    """
    client = get_vector_store(SOURCE_DOC_INDEX_NAME)
    ids = [
        f"company-{ticker}-doc_source-sec-doc_type-10-k-period-{period}"
        for period in periods
        for ticker in tickers
    ]
    print(ids)

    for id in ids:
        # Query Pinecone for similar embeddings
        query_results = client.query(
            vector=NULL_VECTOR,
            include_metadata=True,
            filter={"ref_doc_id": {"$eq": id}},
            top_k=1000,
        )

        # Check the results for duplicates
        duplicates = __find_duplicates(query_results.matches)

        # De-duplicate
        if len(duplicates) == 0:
            logging.info("No duplicates found for %s", id)
        else:
            client.delete(ids=duplicates)
            logging.info(f"Deleted duplicate for %s: %s", id, duplicates)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 dedup_vector_store.py\nDeletes duplicate documents from the vector store."
        )
        sys.exit()
    initialize()
    dedup_documents()
