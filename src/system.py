"""
Initialize stuff
"""
from dotenv import load_dotenv
import logging


def initialize():
    """
    Initialize stuff
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading environment variables from .env file")
    res = load_dotenv(".env")
    if res is False:
        logging.warning("No .env file found")


if __name__ == "__main__":
    initialize()
