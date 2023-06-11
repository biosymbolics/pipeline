import os
import redis  # type: ignore
from typing import Any, Union

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
    value = redis_client().get(key)
    if not value:
        raise Exception(f"Key {key} not found in cache")
    return value


def set_cached_value(key: str, value: Any):
    redis_client().set(key, value, ex=DEFAULT_EXPIRATION)


def get_cached_json(key: str) -> Union[dict, Any]:
    data = redis_client().json().get(key)
    if not data:
        raise Exception(f"Key {key} not found in cache")
    return data


def set_cached_json(key: str, value: Any):
    redis_client().json().set(key, "$", value)
    redis_client().pexpire(key, DEFAULT_EXPIRATION)
