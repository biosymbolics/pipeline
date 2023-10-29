"""
Utils for namespace
"""
from utils.string import get_id
from typings.indices import NamespaceKey


def get_namespace(namespace_key: NamespaceKey) -> str:
    """
    Form namespace from namespace key

    Args:
        namespace_key (NamespaceKey): key of namespace (order matters)
    """
    return "/".join(namespace_key)


def get_namespace_id(namespace_key: NamespaceKey) -> str:
    """
    Form namespace id from namespace key

    Args:
        namespace_key (NamespaceKey): key of namespace (order matters)
    """
    parts = [f"{k}-{get_id(v)}" for k, v in namespace_key._asdict().items()]
    return "-".join(parts)
