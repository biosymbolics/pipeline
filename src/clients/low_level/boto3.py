"""
Boto3 client
"""
import json
import os
from typing import Any, Callable, TypeVar
import boto3
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

T = TypeVar("T")

CACHE_BUCKET = os.environ.get("CACHE_BUCKET", "biosym-patents")


def get_boto_client(service: str):
    """
    Get boto client

    Args:
        service (str): service name (e.g. s3, ssm)
    Returns:
        boto3.client: boto client
    """
    session = boto3.Session(region_name="us-east-1")
    return session.client(service)


def retrieve_with_cache_check(
    operation: Callable[[], T], key: str, cache_name: str = CACHE_BUCKET
) -> T:
    """
    Retrieve data from S3 cache if it exists, otherwise perform the operation and save the result to S3.
    """
    s3 = boto3.client("s3")
    key = f"cache/{key}.json"

    # Check if the result exists in S3
    try:
        logger.info("Checking cache `%s` for key `%s`", cache_name, key)
        response = s3.get_object(Bucket=cache_name, Key=key)
        data = json.loads(response["Body"].read().decode("utf-8"))
        return data
    except s3.exceptions.NoSuchKey:
        logger.info("Checking miss for key: %s", key)
        # If not in cache, perform the operation
        data = operation()

        # Save the result to S3
        s3.put_object(
            Bucket=cache_name,
            Key=key,
            Body=json.dumps(data, default=str),
            ContentType="application/json",
        )

        return data
