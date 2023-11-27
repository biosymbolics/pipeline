"""
Handler for RDS start/stop
"""
import logging
from typing import Literal
from pydantic import BaseModel
from clients.low_level import boto3

from constants.core import DB_CLUSTER

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


Action = Literal["START", "STOP"]


class RdsControlParams(BaseModel):
    action: Action


class RdsControlEvent(BaseModel):
    queryStringParameters: RdsControlParams


rds_client = boto3.get_boto_client("rds")


def update_rds_cluster(action: Action):
    if DB_CLUSTER is None:
        raise ValueError("DB_CLUSTER is not set")

    logger.info(f"Taking action {action} on {DB_CLUSTER}")
    instances = rds_client.describe_db_instances()["DBInstances"]
    logger.info("Found instances: %s", instances)

    if action == "START":
        rds_client.start_db_cluster(DBClusterIdentifier=DB_CLUSTER)
    elif action == "STOP":
        rds_client.stop_db_cluster(DBClusterIdentifier=DB_CLUSTER)
    else:
        raise ValueError(f"Invalid action: {action}")


def rds_control(raw_event: dict, context):
    """
    RDS control handler for stopping/starting RDS cluster

    Invocation:
    - Local: `serverless invoke local --function rds-control --param='ENV=local' --data='{"queryStringParameters": { "action":"START" }}'`
    - Remote: `serverless invoke --function rds-control --data='{"queryStringParameters": { "action":"STOP" }}'`
    """
    event = RdsControlEvent(**raw_event)
    p = event.queryStringParameters

    if not p.action:
        logger.error("Missing param `action`, params: %s", p)
        return {
            "statusCode": 400,
            "body": "Missing parameter(s)",
        }

    try:
        update_rds_cluster(p.action)
    except Exception as e:
        logger.error("Error updating RDS cluster: %s", e)
        return {
            "statusCode": 500,
            "body": "Error updating RDS cluster",
        }

    return {"statusCode": 200, "message": "ok"}
