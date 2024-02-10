"""
Utils related to files
"""

import json
import logging
import os
import pickle
from typing import Any, Union

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def save_as_file(content: Union[str, bytes], filename: str):
    """
    Simple file writer function
    """
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(filename, mode) as file:
        file.write(content)


def load_file(filename: str) -> Any:
    """
    Simple file reader function

    Args:
        filename (str): filename to read
    """
    with open(filename, "r") as file:
        return file.read()


def save_as_pickle(obj: Any, filename: str) -> str:
    """
    Saves obj as pickle
    """
    with open(filename, "wb") as file:
        pickle.dump(obj, file)

    return filename


def maybe_load_pickle(filename: str) -> Any:
    """
    Loads pickle from file if it exists
    """
    if is_pickle_exists(filename):
        try:
            return load_pickle(filename)
        except Exception as e:
            logger.warning("Failure to load pickle %s: %s", filename, e)
            return None
    else:
        return None


def is_pickle_exists(filename: str) -> bool:
    """
    Checks if pickle exists
    """
    return os.path.exists(filename)


def load_pickle(filename: str) -> Any:
    """
    Loads pickle from file
    """
    with open(filename, "rb") as file:
        return pickle.load(file)


def load_pickles(directory: str = "/data/pickles") -> dict[str, Any]:
    """
    Loads all pickles from directory

    Args:
        directory (str): directory to load pickles from

    Returns:
        dict[str, Any]: dict of objects keyed by filename
    """
    pickles = [(filename, load_pickle(filename)) for filename in os.listdir(directory)]
    return dict(pickles)


def save_json_as_file(
    serializable_obj: Any, filename: str, pickle_on_error: bool = True
):
    """
    JSON encodes and persists serializable_obj
    """
    try:
        json_str = json.dumps(serializable_obj, default=str)
        save_as_file(json_str, filename)
    except Exception as e:
        logger.error("Failure to save JSON to file %s: %s", filename, e)
        if pickle_on_error:
            pickle_file = save_as_pickle(serializable_obj, filename)
            logger.error("Saved as pickle: %s", pickle_file)
        raise e


def load_json_from_file(filename: str) -> Any:
    """
    JSON encodes and persists serializable_obj
    """
    json_str = load_file(filename)
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError as e:
        logger.error("Failure to load JSON from file %s", filename)
        raise e
