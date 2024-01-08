"""
Initialize stuff
"""
import os
from dotenv import load_dotenv
import logging

from constants.core import DATABASE_URL


def initialize():
    """
    Initialize stuff
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading environment variables from .env file")
    res = load_dotenv(".env")

    if DATABASE_URL is None:
        raise ValueError("DATABASE_URL not set")

    os.environ["DATABASE_URL"] = DATABASE_URL

    if res is False:
        logging.warning("No .env file found")


if __name__ == "__main__":
    initialize()
