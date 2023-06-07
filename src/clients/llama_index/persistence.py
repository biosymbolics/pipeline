"""
Methods for for llama indexes persistence
"""
import logging
from typing import Optional, Union
from llama_index import (
    load_index_from_storage,
    load_indices_from_storage,
)
from llama_index.indices.base import BaseGPTIndex as LlmIndex

from .constants import DEFAULT_MODEL_NAME
from .context import get_service_context, get_storage_context
from types.indices import LlmModel, NamespaceKey
from .utils import get_persist_dir


def __load_index_or_indices(
    namespace_key: NamespaceKey,
    index_id: Optional[str] = None,
    model_name: Optional[LlmModel] = DEFAULT_MODEL_NAME,
) -> Union[LlmIndex, list[LlmIndex], None]:
    """
    Load persisted index

    Args:
        namespace_key (str): namespace of the index (e.g. SEC-BMY)
        index_id (optional str): unique id of the index (e.g. 2020-01-1).
            all indices loaded if unspecified.
        model_name (optional str): model name to use for index
    """
    try:
        logging.info("Attempting to load index %s/%s", namespace_key, index_id)
        storage_context = get_storage_context(namespace_key)
        service_context = get_service_context(model_name)

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
        print(ex)
        return None


def maybe_load_index(namespace_key: NamespaceKey, index_id: str) -> Optional[LlmIndex]:
    """
    Load index if present, otherwise return none

    Args:
        namespace_key (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    try:
        return load_index(namespace_key, index_id)
    except Exception:
        return None


def load_index(
    namespace_key: NamespaceKey,
    index_id: Optional[str] = None,
    model_name: LlmModel = DEFAULT_MODEL_NAME,
) -> LlmIndex:
    """
    Load persisted index

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (optional str): unique id of the index (e.g. 2020-01-1).
    """
    index = __load_index_or_indices(namespace_key, index_id, model_name=model_name)
    if isinstance(index, list):
        raise Exception("Expected single index, got list")
    if isinstance(index, LlmIndex):
        return index
    raise Exception("Expected single index, got None")


def load_indices(
    namespace_key: NamespaceKey, model_name: LlmModel = DEFAULT_MODEL_NAME
) -> list[LlmIndex]:
    """
    Load persisted indices

    Args:
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
    """
    indices = __load_index_or_indices(namespace_key, model_name=model_name)
    if not isinstance(indices, list):
        raise Exception("Expected list of indices, got single index")
    return indices


def persist_index(index: LlmIndex, namespace_key: NamespaceKey, index_id: str):
    """
    Persist llama index

    Args:
        index (LlmIndex): any generic LLM Index
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    try:
        directory = get_persist_dir(namespace_key)
        index.set_index_id(index_id)
        index.storage_context.persist(persist_dir=directory)
    except Exception as ex:
        logging.error("Error persisting index: %s", ex)
