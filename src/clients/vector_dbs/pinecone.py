"""
Client for pinecone vector db
"""
import os
from typing import Any, Mapping
import pinecone
from llama_index.vector_stores.types import ExactMatchFilter, MetadataFilters

from local_types.indices import NamespaceKey

API_KEY = os.environ["PINECONE_API_KEY"]


def get_metadata_filters(namespace: NamespaceKey) -> MetadataFilters:
    """
    Get metadata filters for namespace

    Args:
        namespace (NamespaceKey): namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
    """
    filters = [
        ExactMatchFilter(key=key, value=value)
        for key, value in namespace._asdict().items()
    ]
    metadata_filters = MetadataFilters(filters=filters)
    return metadata_filters


def get_vector_db(
    index_name: str, pinecone_args: Mapping[str, Any] = {}
) -> pinecone.Index:
    """
    Initializes vector db, creating index if nx

    Args:
        index_name (str): name of index.
        pinecone_args (Mapping[str, Any]): additional args to pass to pinecone.create_index. Defaults to {}.
    """
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
