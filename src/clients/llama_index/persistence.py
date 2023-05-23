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

from .context import get_service_context, get_storage_context
from .utils import get_persist_dir


def __load_index_or_indices(
    namespace: str, index_id: Optional[str] = None
) -> Union[LlmIndex, list[LlmIndex]]:
    """
    Load persisted index

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (optional str): unique id of the index (e.g. 2020-01-1).
            all indices loaded if unspecified.
    """
    try:
        logging.info("Attempting to load index %s/%s", namespace, index_id)
        storage_context = get_storage_context(namespace)
        service_context = get_service_context()

        if index_id:
            index = load_index_from_storage(
                storage_context,
                index_id=index_id,
                service_context=service_context,
            )
            logging.info("Returning index %s/%s from disk", namespace, index_id)
            return index

        indices = load_indices_from_storage(
            storage_context,
            service_context=service_context,
        )
        logging.info("Returning indices %s from disk", namespace)
        return indices

    except Exception as ex:
        logging.info("Failed to load %s/%s from disk: %s", namespace, index_id, ex)
        print(ex)
        return None


def load_index(namespace: str, index_id: str) -> LlmIndex:
    """
    Load persisted index

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (optional str): unique id of the index (e.g. 2020-01-1).
                                 all indices loaded if unspecified.
    """
    return __load_index_or_indices(namespace, index_id)


def load_indices(namespace: str) -> list[LlmIndex]:
    """
    Load persisted indices

    Args:
        namespace (str): namespace of the index (e.g. SEC-BMY)
    """
    return __load_index_or_indices(namespace)


def persist_index(index: LlmIndex, namespace: str, index_id: str):
    """
    Persist llama index

    Args:
        index (LlmIndex): any generic LLM Index
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    try:
        directory = get_persist_dir(namespace)
        index.set_index_id(index_id)
        index.storage_context.persist(persist_dir=directory)
    except Exception as ex:
        logging.error("Error persisting index: %s", ex)
