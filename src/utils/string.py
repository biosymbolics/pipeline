"""
String utilities
"""


from datetime import date
from functools import reduce
import regex as re
from spacy.tokens import Doc
from typing import Mapping, TypeGuard, Union


_Idable = str | list[str] | int | date
Idable = _Idable | Mapping[str, _Idable]


def get_id(value: Idable) -> str:
    """
    Returns the id of a value

    Args:
        string (str or list[str]): string to get id of
    """
    if isinstance(value, Mapping):
        value = "_".join(
            [
                f"{key}={get_id(value or '')}"
                for key, value in sorted(value.items(), key=lambda item: item[0])
            ]
        )

    if isinstance(value, list):
        value = "_".join(value)

    if isinstance(value, bool):
        value = str(value)

    if isinstance(value, date):
        value = value.isoformat()

    if isinstance(value, int):
        value = str(value)

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


def generate_ngrams(tokens: Doc, n: int) -> list[tuple[tuple[str, ...], list[float]]]:
    """
    Generate n-grams (term & token) from a list of Spacy tokens

    Args:
        tokens (Doc): list of tokens
        n (int): n-gram size

    Returns:
        list[tuple[tuple[str, str], list[float]]]: list of n-grams tuples and their vectors
    """
    index_sets: list[tuple[int, ...]] = reduce(
        lambda acc, i: acc + [(i, *[i + grm + 1 for grm in range(n - 1)])],
        range(len(tokens) + 1 - n),
        [],
    )
    ngrams = [tuple(tokens[i].text for i in iset) for iset in index_sets]
    vectors = [list(tokens[min(iset) : max(iset)].vector) for iset in index_sets]
    return list(zip(ngrams, vectors))


def generate_ngram_phrases(tokens: Doc, n: int) -> list[tuple[str, list[float]]]:
    """
    Generate n-grams (term & token) from a list of Spacy tokens

    Args:
        tokens (Doc): list of tokens
        n (int): n-gram size

    Returns:
        list[tuple[str, list[float]]]: list of n-grams and their vectors
    """
    return [(" ".join(ng[0]), ng[1]) for ng in generate_ngrams(tokens, n)]
