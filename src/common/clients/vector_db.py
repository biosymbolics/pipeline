"""
Client for vector database (currently pinecone)
TODO: not yet working
"""
import os
import pinecone
from llama_index.indices.base import BaseGPTIndex as LlmIndex
from multipledispatch import dispatch


API_KEY = os.environ["PINECONE_API_KEY"]


@dispatch(str)  # type: ignore[no-redef]
def init_vector_db(index_name: str) -> pinecone.Index:
    """
    Initializes vector db, creating index if nx

    Args:
        index_name (str): name of index (will correspond with "namespace" used elsewhere) (e.g. SEC-BMY)
    """
    pinecone.deinit()
    pinecone.init(api_key=API_KEY)  # environment/datacenter?

    if index_name not in pinecone.list_indexes():
        return pinecone.create_index(
            index_name, metric="cosine", shards=1
        )  # dimension=1536, pod_type="p1"

    return pinecone.Index(f"{index_name}-index")


@dispatch(str, list[object])  # type: ignore[no-redef]
def init_vector_db(index_name: str, indices: list[LlmIndex]):
    """
    Initializes vector db; adds indices

    Args:
        index_name (str): name of index (will correspond with "namespace" used elsewhere) (e.g. SEC-BMY)
        indicies (list[LlmIndex]): vectors
    """
    vector_db = init_vector_db(index_name)
    vector_db.upsert(vectors={indices})
