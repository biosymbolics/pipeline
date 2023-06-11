"""
Cache client
"""
from typing import Any, Callable
import json
from hashlib import md5
import logging

from .redis import get_cached_value, set_cached_value


def __get_cache_key(identifiers: list) -> str:
    """
    Creates a cache key by hasing the identifier(s) provided

    For example:
    ```
    __get_cache_key(["PFE", "SEC", "10-K", "2020-01-01"]) -> "1df051d5fc67f9192bdc7d77957d9875"
    ```

    Args:
        identifiers (list): list of identifiers to use for cache key
    """
    string = "-".join([json.dumps(id) for id in identifiers])
    key = (md5(string.encode())).hexdigest()
    return key


def get_from_cache(identifiers: list) -> Any:
    """
    Get a cached object. Returns None if not found.

    Args:
        identifiers (list): list of identifiers to use for cache key
    """
    key = __get_cache_key(identifiers)

    try:
        cached = get_cached_value(key)
    except KeyError:
        cached = None  # not found in cache

    if cached:
        logging.info("Cache hit for %s", key)

    return cached


def set_in_cache(identifiers: list, object: Any) -> None:
    """
    Sets in cache

    Args:
        identifiers (list): list of identifiers to use for cache key
        object (Any): object to set in cache
    """
    key = __get_cache_key(identifiers)
    logging.info("Setting value in cache: %s", key)
    set_cached_value(key, object)


def call_with_cache(identifiers: list, fetcher: Callable[[], Any]) -> Any:
    """
    Get from cache, or call provided method to fetch & set in cache

    Args:
        identifiers (list): list of identifiers to use for cache key
        fetcher (Callable[[], Any]): method to call if not found in cache
    """
    from_cache = get_from_cache(identifiers)
    if from_cache:
        return from_cache

    new_result = fetcher()
    set_in_cache(identifiers, new_result)

    return new_result
