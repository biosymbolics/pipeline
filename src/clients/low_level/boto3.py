"""
Boto3 client
"""
import boto3


def get_boto_client(service: str):
    """
    Get boto client

    Args:
        service (str): service name (e.g. s3, ssm)
    Returns:
        boto3.client: boto client
    """
    session = boto3.Session()
    return session.client(service)
