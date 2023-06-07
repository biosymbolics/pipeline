"""
Methods for for llama indexes persistence
"""
import logging
from typing import Optional, Union
from llama_index import (
    load_index_from_storage,
    load_indices_from_storage,
)

from types.indices import LlmIndex, NamespaceKey
from .context import (
    DEFAULT_CONTEXT_ARGS,
    ContextArgs,
    get_service_context,
    get_storage_context,
)
from .utils import get_persist_dir


def __load_some_indices(
    namespace_key: NamespaceKey,
    index_id: Optional[str] = None,
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> Union[LlmIndex, list[LlmIndex], None]:
    """
    Load persisted index

    Args:
        namespace_key (str): namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        index_id (optional str): unique id of the index (e.g. 2020-01-1).
            all indices loaded if unspecified.
        context_args (ContextArgs): context args for loading index
    """
    try:
        logging.info("Attempting to load index %s/%s", namespace_key, index_id)
        storage_context = get_storage_context(
            namespace_key, **context_args.storage_args
        )
        service_context = get_service_context(model_name=context_args.model_name)

        if index_id:
            index = load_index_from_storage(
                storage_context,
                index_id=index_id,
                service_context=service_context,
            )
            logging.info("Returning index %s/%s from disk", namespace_key, index_id)
            return index

        indices = load_indices_from_storage(
            storage_context,
            service_context=service_context,
        )
        logging.info("Returning indices %s from disk", namespace_key)
        return indices

    except Exception as ex:
        logging.info("Failed to load %s/%s from disk: %s", namespace_key, index_id, ex)
        return None


def load_index(
    namespace_key: NamespaceKey,
    index_id: Optional[str] = None,
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> LlmIndex:
    """
    Load persisted index

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        index_id (str): unique id of the index (e.g. 2020-01-1).
        context_args (ContextArgs): context args for loading index
    """
    index = __load_some_indices(namespace_key, index_id, context_args)
    if isinstance(index, list):
        raise Exception("Expected single index, got list")
    if isinstance(index, LlmIndex):
        return index
    raise Exception("Expected single index, got None")


def load_indices(
    namespace_key: NamespaceKey,
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> list[LlmIndex]:
    """
    Load persisted indices

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
    """
    indices = __load_some_indices(namespace_key, None, context_args)
    if not isinstance(indices, list):
        raise Exception("Expected list of indices, got single index")
    return indices


def persist_index(index: LlmIndex, namespace_key: NamespaceKey, index_id: str):
    """
    Persist llama index

    Args:
        index (LlmIndex): any generic LLM Index
        namespace_key (NamespaceKey) namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    try:
        directory = get_persist_dir(namespace_key)
        index.set_index_id(index_id)
        index.storage_context.persist(persist_dir=directory)
    except Exception as ex:
        logging.error("Error persisting index: %s", ex)


def maybe_load_index(
    namespace_key: NamespaceKey,
    index_id: Optional[str] = None,
    context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
) -> Optional[LlmIndex]:
    """
    Load index if present, otherwise return none

    Args:
        namespace_key (str): namespace of the index (e.g. (company="BIBB", doc_source="SEC", doc_type="10-K"))
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    try:
        return load_index(namespace_key, index_id, context_args)
    except Exception:
        return None
