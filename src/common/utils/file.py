"""
Utils related to files
"""
import json
import pickle
from typing import Any, Union


def save_as_file(content: Union[str, bytes], filename: str):
    """
    Simple file writer function
    """
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(filename, mode) as file:
        file.write(content)


def save_as_pickle(obj: Any, filename: str):
    """
    Saves obj as picke
    """
    with open(filename, "wb") as file:
        # Pickle the object
        pickle.dump(obj, file)


def save_json_as_file(serializable_obj: Any, filename: str):
    """
    JSON encodes and persists serializable_obj
    """
    json_str = json.dumps(serializable_obj)
    save_as_file(json_str, filename)
