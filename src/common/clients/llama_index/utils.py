import pathlib

from common.clients.llama_index.constants import BASE_STORAGE_DIR


def get_persist_dir(namespace: str) -> str:
    directory = f"{BASE_STORAGE_DIR}/{namespace}"
    pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    return directory
