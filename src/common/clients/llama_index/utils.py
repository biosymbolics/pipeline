from common.clients.llama_index.constants import BASE_STORAGE_DIR


def get_persist_dir(namespace: str) -> str:
    return f"{BASE_STORAGE_DIR}/{namespace}"
