"""
Initialize stuff
"""
from dotenv import load_dotenv
import logging


def init():
    """
    Initialize stuff
    """
    logging.basicConfig(level=logging.INFO)

    logging.info("Loading environment variables from .env file")
    load_dotenv("/Users/kristinlindquist/development/pipeline/.env")


if __name__ == "__main__":
    init()
