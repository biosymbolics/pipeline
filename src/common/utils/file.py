"""
Utils related to files
"""
import json
import pickle
import uuid
from typing import Any, Optional, Union


def save_as_file(content: Union[str, bytes], filename: str):
    """
    Simple file writer function
    """
    mode = "wb" if isinstance(content, bytes) else "w"
    with open(filename, mode) as file:
        file.write(content)


def save_as_pickle(
    obj: Any, filename: Optional[str] = None, use_uuid: bool = True
) -> str:
    """
    Saves obj as pickle
    """
    PICKLE_BASE = "data/pickles/"
    if not filename or use_uuid:
        filename = PICKLE_BASE + (filename or "") + str(uuid.uuid4()) + ".txt"
    with open(filename, "wb") as file:
        pickle.dump(obj, file)

    return filename


def save_json_as_file(serializable_obj: Any, filename: str):
    """
    JSON encodes and persists serializable_obj
    """
    json_str = json.dumps(serializable_obj)
    save_as_file(json_str, filename)
