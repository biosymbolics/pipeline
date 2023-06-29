"""
String utilities
"""


import re
from typing import Union


def get_id(string: Union[str, list[str]]) -> str:
    """
    Returns the id of a string

    Args:
        string (str or list[str]): string to get id of
    """
    if isinstance(string, list):
        string = "_".join(string)

    return string.replace(" ", "_").lower()


def remove_comment_syntax(text: str) -> str:
    """
    Remove leading ```json and trailing ``` (and anything after it)
    (used for parsing LlamaIndex/Langchain/GPT answers)

    TODO: replace with langchain `parse_json_markdown`

    Example:
        >>> obj_str = __remove_comment_syntax('```json\n{"k01":"t1","k02":"t2"}``` ```json\n{"k11":"t1","k12":"t2"},{"k21":"t1","k22":"t2"}```')
        >>> json.loads(obj_str)
        {'k11': 't1', 'k12': 't2'}, {'k21': 't1', 'k22': 't2'}
    """
    json_blocks = re.findall(r"\s*```json(.*?)```", text, re.DOTALL)
    if len(json_blocks) == 0:
        return text
    elif len(json_blocks) > 1:
        return json_blocks[-1]  # return the last

    return json_blocks[0]


def chunk_text(content: str, chunk_size: int) -> list[str]:
    """
    Turns a list into a list of lists of size `batch_size`

    Args:
        content (str): content to chunk
        chunk_size (int): chunk size
    """
    return [content[i : i + chunk_size] for i in range(0, len(content), chunk_size)]


def chunk_list(content_list: list[str], chunk_size: int) -> list[list[str]]:
    """
    Chunk a list of content

    Args:
        content (list): content to chunk
        chunk_size (int): chunk size
    """
    return [chunk_text(content, chunk_size) for content in content_list]
