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


def is_file_exists(filename: str, path: str = "") -> bool:
    """
    Checks if file exists
    """
    file_path = os.path.join(path, filename)
    return os.path.exists(file_path)


def save_as_file(content: Union[str, bytes], filename: str, path: str = "") -> str:
    """
    Simple file writer function
    """
    if path and not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, filename)

    mode = "wb" if isinstance(content, bytes) else "w"
    with open(file_path, mode) as file:
        file.write(content)

    return file_path


def load_file(filename: str, path: str = "") -> Any:
    """
    Simple file reader function

    Args:
        filename (str): filename to read
    """
    file_path = os.path.join(path, filename)

    if not is_file_exists(filename, path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "r") as file:
        return file.read()


def save_as_pickle(obj: Any, filename: str, path: str = "") -> str:
    """
    Saves obj as pickle
    """
    if path and not os.path.exists(path):
        os.makedirs(path)

    file_path = os.path.join(path, filename)

    with open(file_path, "wb") as file:
        pickle.dump(obj, file)

    return file_path


def maybe_load_pickle(filename: str, path: str = "") -> Any:
    """
    Loads pickle from file if it exists
    """
    try:
        return load_pickle(filename, path)
    except Exception as e:
        logger.warning("Failure to load pickle %s: %s", filename, e)
        return None


def load_pickle(filename: str, path: str = "") -> Any:
    """
    Loads pickle from file
    """
    file_path = os.path.join(path, filename)

    if not is_file_exists(filename, path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, "rb") as file:
        return pickle.load(file)


def save_json_as_file(
    serializable_obj: Any, filename: str, path: str = "", pickle_on_error: bool = True
) -> str:
    """
    JSON encodes and persists serializable_obj
    """
    try:
        json_str = json.dumps(serializable_obj, default=str)
        file_path = save_as_file(json_str, filename, path)
        return file_path
    except Exception as e:
        logger.error("Failure to save JSON to file %s: %s", filename, e)
        if pickle_on_error:
            pickle_file = save_as_pickle(serializable_obj, filename, path)
            logger.error("Saved as pickle: %s", pickle_file)
        raise e


def load_json_from_file(filename: str, path: str = "") -> Any:
    """
    JSON encodes and persists serializable_obj
    """
    json_str = load_file(filename, path)
    try:
        return json.loads(json_str)
    except json.decoder.JSONDecodeError as e:
        logger.error("Failure to load JSON from file %s", filename)
        raise e
