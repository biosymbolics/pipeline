"""
Utils for namespace
"""
from types.indices import NamespaceKey


def get_namespace(namespace_key: NamespaceKey) -> str:
    """
    Form namespace from namespace key

    Args:
        namespace_key (list[str]): key of namespace (order matters)
    """
    return "/".join(namespace_key)
