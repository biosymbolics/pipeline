"""
Utils for namespace
"""
from common.utils.string import get_id
from local_types.indices import NamespaceKey


def get_namespace(namespace_key: NamespaceKey) -> str:
    """
    Form namespace from namespace key

    Args:
        namespace_key (list[str]): key of namespace (order matters)
    """
    return "/".join(namespace_key)


def get_namespace_id(namespace_key: NamespaceKey) -> str:
    """
    Form namespace id from namespace key

    Args:
        namespace_key (list[str]): key of namespace (order matters)
    """
    parts = [f"{k}-{get_id(v)}" for k, v in namespace_key._asdict()]
    return "-".join(parts)
