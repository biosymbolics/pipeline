"""
LlamaIndex utils
"""
import pathlib

from clients.llama_index.constants import BASE_STORAGE_DIR
from types.indices import NamespaceKey


def get_namespace(namespace_key: NamespaceKey) -> str:
    """
    Form namespace from namespace key

    Args:
        namespace_key (list[str]): key of namespace (order matters)
    """
    return "/".join(namespace_key)


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
