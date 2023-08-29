import os
from pathlib import Path
import logging

from clients.low_level.boto3 import fetch_s3_obj

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DEFAULT_LOCATION = os.path.join(os.getcwd(), "models")


def get_model_path(model: str, dest: str = DEFAULT_LOCATION) -> str:
    """
    Get model path from disk, downloading from S3 if necessary
    """

    filename = os.path.join(Path(dest), model)
    logger.info(f"Getting model from disk (%s)", filename)

    if not os.path.exists(filename):
        logger.info("Not in disk, downloading from S3 bucket")
        filename = download_model_from_s3(model, dest)
        return filename

    return filename


def download_model_from_s3(model: str, dest: str = DEFAULT_LOCATION) -> str:
    """
    Download model from S3
    """
    logger.info("Downloading %s from S3", model)
    filename = os.path.join(Path(dest), model)
    object_name = f"models/{model}"  # targz?
    fetch_s3_obj(object_name, filename)

    return filename
