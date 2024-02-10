"""
Text classifiers
"""

from enum import Enum
import regex as re
from pydash import compact, flatten, uniq
from typing import Mapping, TypeVar
import logging

from core.ner.utils import lemmatize_all
from utils.re import expand_res
from utils.string import generate_ngrams

T = TypeVar("T", bound=Enum | str)


def create_lookup_map(keyword_map: Mapping[T, list[str]]) -> Mapping[str, T]:
    """
    Create a lookup map from a keyword map, with keywords lemmatized.

    Expands all regexes (and thus will barf on infinity regex like "blah .*")

    Lower-cases all keywords.

    Usage:
    ```
        >>> keyword_map = {
            "INDICATION": ["indication", "diseases"],
            "INTERVENTION": ["drug", "compounds"],
        }
        >>> create_lookup_map(keyword_map)
        # yields lemmatized keyword to category mapping:
        {
            'indication': 'INDICATION',
            'disease': 'INDICATION',
            'drug': 'INTERVENTION',
            'compound': 'INTERVENTION',
        }
        >>>
    ```
    """
    cat_key_tups = [
        (keyword.lower(), category)
        for category, keywords in keyword_map.items()
        for keyword in flatten(expand_res(keywords))
    ]
    lookup_tups = [(lemmatize_all(tup[0]), tup[1]) for tup in cat_key_tups]
    lookup_map = dict(lookup_tups)
    logging.debug(f"Created lookup map: %s", lookup_map)
    return lookup_map


def classify_string(
    string: str, lookup_map: Mapping[str, T], nx_name: T | None = None
) -> list[T]:
    """
    Classify a string by keywords + lookup map.
    Assumes lookup map has lemmatized keywords.
    Sorts for consistency.
    Looks at bi- and tri-grams.
    Lower-cases all keywords.

    Args:
        string (str): string to classify
        lookup_map (Mapping[str, str]): mapping of lemmatized keywords to categories
        nx_name (str): name to use for non-matches
    """

    def _generate_ngrams(_string: str):
        lemmatized = lemmatize_all(_string.lower())
        tokens = tuple(lemmatized.split(" "))
        bigrams = generate_ngrams(tokens, 3)
        trigrams = generate_ngrams(tokens, 2)
        ngrams = [
            *tokens,
            *[" ".join(ng) for ng in [*bigrams, *trigrams]],
        ]

        # remove trailing punct
        return [re.sub(r"[.,]$", "", ngram) for ngram in ngrams]

    # generate ngrams
    ngrams = _generate_ngrams(string)
    categories = [lookup_map.get(ngram, nx_name) for ngram in ngrams]
    return sorted(uniq(compact(categories)))  # type: ignore


def classify_by_keywords(
    strings: list[str],
    keyword_map: Mapping[T, list[str]],
    nx_name: T | None = None,
) -> list[list[T]]:
    """
    Classify a string by keywords + keyword map.
    Uses lemmatization.
    Output indexed by input

    Args:
        strings (list[str]): strings to classify
        keyword_map (Mapping[str, list[str]]): mapping of categories to keywords
        nx_name (T): name to use for non-matches

    Returns: a list of categories

    Example:
        >>> classify_by_keywords(pl.Series(["Method of treatment by tryptamine alkaloids"]), get_patent_attribute_map())
    """
    lookup = create_lookup_map(keyword_map)

    return [classify_string(s, lookup_map=lookup, nx_name=nx_name) for s in strings]
