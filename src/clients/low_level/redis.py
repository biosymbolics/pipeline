import os
import redis  # type: ignore
from typing import Any, Union

from typings import JsonSerializable

REDIS_ENDPOINT = os.environ.get("REDIS_HOST")

DEFAULT_EXPIRATION = 1000 * 60 * 60 * 1  # 1 hour


class RedisClient:
    """
    Class for Redis client (lazy init improves cold start perf)
    """

    def __init__(self):
        if not REDIS_ENDPOINT:
            raise Exception("REDIS_HOST environment variable not set")
        self.endpoint = REDIS_ENDPOINT
        self.client = None

    def get_client(self):
        return redis.Redis(self.endpoint)

    def __call__(self):
        if not self.client:
            self.client = self.get_client()
        return self.client


redis_client = RedisClient()


def get_cached_value(key: str) -> Any:
    """
    Get a cached value. Raises KeyError if not found.

    Args:
        key (str): key to use for cache lookup
    """
    value = redis_client().get(key)
    if not value:
        raise KeyError(f"Key {key} not found in cache")
    return value


def set_cached_value(key: str, value: Union[str, bytes, float, int]):
    """
    Sets value in cache

    Args:
        key (str): key to use for cache lookup
        value (Union[str, bytes, float, int]): value to set in cache. Should be of a type
                    that can be serialized and stored in Redis (str, bytes, float, or int).
                    Use set_cached_json for other types.
    """
    redis_client().set(key, value, ex=DEFAULT_EXPIRATION)


def get_cached_json(key: str) -> JsonSerializable:
    """
    Get a cached JSON object. Raises KeyError if not found.

    Args:
        key (str): key to use for cache lookup
    """
    data = redis_client().json().get(key)
    if not data:
        raise KeyError(f"Key {key} not found in cache")
    return data


def set_cached_json(key: str, value: JsonSerializable):
    """
    Sets JSON value in cache

    Args:
        key (str): key to use for cache lookup
        value (JsonSerializable): value to set in cache. Should be of a type
    """
    redis_client().json().set(key, "$", value)
    redis_client().pexpire(key, DEFAULT_EXPIRATION)
