"""
String utilities
"""

from datetime import date
from enum import Enum
import json
from pydantic import BaseModel
import regex as re
from typing import Mapping, Sequence, TypeGuard, Union
from hashlib import sha1


_IdablePrimitives = str | int | date | float | list[str] | None
_Idable = _IdablePrimitives | Sequence[_IdablePrimitives]
Idable = _Idable | Mapping[str, _Idable] | Sequence[_Idable]


MAX_ID_CHARS = 200


def create_hash_key(identifiers: list) -> str:
    """
    Creates a hash key by hasing the identifier(s) provided

    For example:
    ```
    create_hash_key(["PFE", "SEC", "10-K", "2020-01-01"]) -> "1df051d5fc67f9192bdc7d77957d9875"
    ```

    Args:
        identifiers (list): list of identifiers to use for cache key
    """
    string = "-".join([json.dumps(id) for id in identifiers])
    key = (sha1(string.encode())).hexdigest()
    return key


def get_id(value: Idable) -> str:
    """
    Returns the id of a value

    Args:
        string (str or list[str]): string to get id of
    """
    if value is None:
        return ""

    if isinstance(value, BaseModel):
        return get_id(value.model_dump())

    if isinstance(value, Mapping):
        value = "_".join(
            [
                f"{key}={get_id(value or '')}"
                for key, value in sorted(value.items(), key=lambda item: item[0])
            ]
        )

    if isinstance(value, list) or (
        isinstance(value, Sequence) and not isinstance(value, str)
    ):
        value = "_".join([str(v) for v in value])
        if len(value) > MAX_ID_CHARS:
            # create a hash key if otherwise too long
            # i.e. when we're looking up docs by a list of ids
            value = create_hash_key([str(v) for v in value])

    if isinstance(value, bool):
        value = str(value)

    if isinstance(value, date):
        value = value.isoformat()

    if isinstance(value, int) or isinstance(value, float):
        value = str(value)

    if isinstance(value, Enum):
        value = str(value.value)

    return value.replace(" ", "_").lower()


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
    json_blocks = re.findall(
        r"\s*(?:```)+[a-z]{4,}(?:\s|\n|\b)(.*?)(?:```)+", text, re.DOTALL
    )
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


def is_bytes_dict(maybe_byte_dict) -> TypeGuard[dict[bytes, bytes]]:
    """
    Checks if a dictionary is a dictionary of bytes
    """
    return isinstance(maybe_byte_dict, dict) and all(
        isinstance(key, bytes) and isinstance(value, bytes)
        for key, value in maybe_byte_dict.items()
    )


def is_strings_dict(maybe_strings_dict) -> TypeGuard[dict[str, str]]:
    """
    Checks if a dictionary is a dictionary of strings
    """
    return isinstance(maybe_strings_dict, dict) and all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in maybe_strings_dict.items()
    )


def byte_dict_to_string_dict(
    maybe_byte_dict: Union[Mapping[bytes, bytes], Mapping[str, str]]
) -> Mapping[str, str]:
    """
    Converts a dictionary of bytes to a dictionary of strings

    Args:
        maybe_byte_dict (dict[bytes, bytes]): dictionary of bytes or strings
    """
    if is_bytes_dict(maybe_byte_dict):
        return dict(
            map(
                lambda item: (item[0].decode(), item[1].decode()),
                maybe_byte_dict.items(),
            )
        )

    if is_strings_dict(maybe_byte_dict):
        return maybe_byte_dict

    raise ValueError(
        f"Expected a dictionary of bytes or strings, got {type(maybe_byte_dict)}"
    )


# TODO: get rid of these, or combine with linker variants
def generate_ngrams(tokens: tuple[str, ...], n: int) -> list[tuple[str, str]]:
    """
    Generate n-grams from a list of tokens

    Args:
        tokens (tuple[str]): list of tokens
        n (int): n-gram size

    Example:
        >>> generate_ngrams(["a", "b", "c", "d"], 2)
        [('a', 'b'), ('b', 'c'), ('c', 'd')]
    """
    return list(zip(*[tokens[i:] for i in range(n)]))  # type: ignore


def generate_ngram_phrases(tokens: tuple[str, ...], n: int) -> list[str]:
    """
    Generate n-grams from a list of tokens

    Args:
        tokens (tuple[str]): list of tokens
        n (int): n-gram size

    Example:
        >>> generate_ngram_phrases(["a", "b", "c", "d"], 2)
        ['a b', 'b c', 'c d']
    """
    return [" ".join(ng) for ng in generate_ngrams(tokens, n)]
