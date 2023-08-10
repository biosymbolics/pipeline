"""
LlamaIndex utils
"""
import pathlib

from clients.llama_index.constants import BASE_STORAGE_DIR
from utils.namespace import get_namespace
from typings.indices import NamespaceKey


def get_persist_dir(namespace_key: NamespaceKey) -> str:
    """
    Get directory for persisting indices

    Args:
        namespace_key (list[str]): key of namespace (order matters)
    """
    namespace = get_namespace(namespace_key)
    directory = f"{BASE_STORAGE_DIR}/{namespace}"
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return directory
