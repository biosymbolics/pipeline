"""
Client for pinecone vector db
"""
import os
from typing import Any, Mapping
import pinecone

from common.utils.namespace import get_namespace
from types.indices import NamespaceKey

API_KEY = os.environ["PINECONE_API_KEY"]


def init_vector_db(
    namespace_key: NamespaceKey, pinecone_args: Mapping[str, Any] = {}
) -> pinecone.Index:
    """
    Initializes vector db, creating index if nx

    Args:
        index_name (str): name of index. Corresponds with "namespace", e.g. ("BIBB", "SEC", "10-K")
        pinecone_args (Mapping[str, Any]): additional args to pass to pinecone.create_index. Defaults to {}.
    """
    index_name = get_namespace(namespace_key)
    pinecone.init(api_key=API_KEY)

    if index_name not in pinecone.list_indexes():
        pinecone_index = pinecone.create_index(
            index_name,
            metric="cosine",
            shards=1,
            dimension=1536,
            **pinecone_args,
        )
        if not pinecone_index:
            raise Exception("Could not create index")

        return pinecone_index

    return pinecone.Index(f"{index_name}-index")
