"""
Methods for for llama indexes persistence
"""
import logging
from typing import Optional
from llama_index import (
    GPTVectorStoreIndex,
)

from local_types.indices import LlmIndex
from .context import (
    DEFAULT_CONTEXT_ARGS,
    ContextArgs,
    get_storage_context,
)


def maybe_load_index(
    index_name: str,
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> Optional[LlmIndex]:
    """
    Load index if present, otherwise return none

    Args:
        index_name (str): name of the index
        context_args (ContextArgs): context args for loading index
    """
    try:
        return load_index(index_name, context_args)
    except Exception:
        return None


def load_index(
    index_name: str,
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> LlmIndex:
    """
    Load persisted index

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        index_id (str): unique id of the index (e.g. 2020-01-1).
        context_args (ContextArgs): context args for loading index
    """
    storage_context = get_storage_context(index_name, **context_args.storage_args)

    index = GPTVectorStoreIndex([], storage_context=storage_context)
    logging.info("Returning vector index %s", index_name)

    if not index:
        raise Exception(f"Index {index_name} not found")
    return index
