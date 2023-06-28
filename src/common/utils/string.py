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
