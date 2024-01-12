"""
Boto3 client
"""
import json
import os
from typing import Any, Awaitable, Callable, Sequence, TypeVar, cast
import boto3
from botocore.exceptions import ClientError
import logging

from utils.date import date_deserializer
from utils.encoding.json_encoder import StorageDataclassJSONEncoder

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T = TypeVar("T", bound=Sequence)

DEFAULT_BUCKET = os.environ.get("CACHE_BUCKET", "biosym-patents")
REGION = os.environ.get("AWS_REGION", "us-east-1")


def get_boto_client(service: str):
    """
    Get boto client

    Args:
        service (str): service name (e.g. s3, ssm)
    Returns:
        boto3.client: boto client
    """
    session = boto3.Session(region_name=REGION)
    return session.client(service)


def fetch_s3_obj(
    key: str, new_filename: str | None = None, bucket: str = DEFAULT_BUCKET
) -> Any:
    """
    Fetch S3 object
    """
    s3 = get_boto_client("s3")

    if new_filename is not None:
        logger.info("Downloading S3 object `%s` to `%s`", key, new_filename)
        s3.download_file(bucket, key, new_filename)
        return

    response = s3.get_object(Bucket=bucket, Key=key)
    return json.loads(response["Body"].read().decode("utf-8"))


def get_cache_key(key, is_all: bool | None = None, limit: int | None = None) -> str:
    """
    Get cache key depending upon whether the result is all or not.

    Args:
        key (str): key
        is_all (bool): whether the result is all or not
        limit (int): limit
    """
    if limit is None:
        return f"cache/{key}.json"

    if is_all:
        return f"cache/{key}_limit=all.json"

    return f"cache/{key}_limit={limit}.json"


def get_cache_with_all_check(
    key: str, limit: int, cache_name: str = DEFAULT_BUCKET
) -> Any:
    """
    Get results from cache with all check.

    First see if a limit=all result exists. If so, return.
    Otherwise, attempt to get the result with limit=limit. Throws exception if not found.
    """
    s3 = get_boto_client("s3")
    try:
        all_cache_key = get_cache_key(key, is_all=True, limit=limit)
        return s3.get_object(Bucket=cache_name, Key=all_cache_key)
    except s3.exceptions.NoSuchKey:
        limit_cache_key = get_cache_key(key, is_all=False, limit=limit)
        return s3.get_object(Bucket=cache_name, Key=limit_cache_key)


def storage_decoder(obj: Any) -> Any:
    return json.loads(obj, object_hook=date_deserializer)


async def retrieve_with_cache_check(
    operation: Callable[[int], Awaitable[T]] | Callable[[], Awaitable[T]],
    key: str,
    limit: int | None = None,
    cache_name: str = DEFAULT_BUCKET,
    encode: Callable[[T], str | bytes] = StorageDataclassJSONEncoder().encode,
    decode: Callable[[str | bytes], T] = storage_decoder,
) -> T:
    """
    Retrieve data from S3 cache if it exists, otherwise perform the operation and save the result to S3.

    Args:
        operation (Callable[[int], T] | Callable[[], T]): operation to perform
        key (str): key
        limit (int): limit
        cache_name (str): cache name
        encode (Callable[[T], str | bytes]): encoder function
    """
    s3 = get_boto_client("s3")

    try:
        logger.info("Checking cache `%s` for key `%s`", cache_name, key)

        if limit is not None:
            # If limit is set, first check if there is a limit=all result.
            response = get_cache_with_all_check(key, limit, cache_name=cache_name)
        else:
            response = s3.get_object(Bucket=cache_name, Key=get_cache_key(key))

        data = decode(response["Body"].read().decode("utf-8"))

        if limit is not None and len(data) > limit:
            return data[0:limit]  # type: ignore
        return data
    except ClientError as ex:
        if not ex.response["Error"]["Code"] == "NoSuchKey":
            raise ex

        logger.info("Checking miss for key: %s", key)

        # If not in cache, perform the operation
        if limit:
            data: T = await operation(limit=limit)  # type: ignore
        else:
            data: T = await operation()  # type: ignore

        # if limit is set and result is list, see if the result size is less than limit.
        # if so, adjust the cache key to indicate that this is all the results.
        is_all = limit is not None and isinstance(data, list) and len(data) < limit
        cache_key = get_cache_key(key, is_all=is_all, limit=limit)

        s3.put_object(
            Bucket=cache_name,
            Key=cache_key,
            Body=encode(data),  # type: ignore
            ContentType="application/json",
        )

        return cast(T, data)
