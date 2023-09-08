"""
Text classifiers
"""
import re
from pydash import compact
from typing import Mapping, Union
import logging

from core.ner.utils import lemmatize_all
from utils.list import dedup


def create_lookup_map(keyword_map: Mapping[str, list[str]]) -> Mapping[str, str]:
    """
    Create a lookup map from a keyword map, with keywords lemmatized.

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
        (keyword, category)
        for category, keywords in keyword_map.items()
        for keyword in keywords
    ]
    lookup_tups = [(lemmatize_all(tup[0]), tup[1]) for tup in cat_key_tups]
    lookup_map = dict(lookup_tups)
    logging.debug(f"Created lookup map: %s", lookup_map)
    return lookup_map


def generate_ngrams(input_list: list[str], n: int):
    return zip(*[input_list[i:] for i in range(n)])


def classify_string(
    string: str, lookup_map: Mapping[str, str], nx_name: Union[str, None] = None
) -> list[str]:
    """
    Classify a string by keywords + lookup map.
    Assumes lookup map has lemmatized keywords.
    Sorts for consistency.
    Looks at most for 3-grams.

    Args:
        string (str): string to classify
        lookup_map (Mapping[str, str]): mapping of lemmatized keywords to categories
        nx_name (str): name to use for non-matches
    """

    def __extract_ngrams():
        lemmatized = lemmatize_all(string.lower())
        tokens = lemmatized.split(" ")
        bigrams = generate_ngrams(tokens, 3)
        trigrams = generate_ngrams(tokens, 2)
        ngrams = [
            *tokens,
            *[" ".join(ng) for ng in [*bigrams, *trigrams]],
        ]

        no_trailing_punct = [re.sub(r"[.,]$", "", ngram) for ngram in ngrams]
        return no_trailing_punct

    ngrams = __extract_ngrams()
    categories = [
        lookup_map.get(re.sub(r"[.,]$", "", ngram), nx_name) for ngram in ngrams
    ]
    return sorted(dedup(compact(categories)))


def classify_by_keywords(
    strings: list[str],
    keyword_map: Mapping[str, list[str]],
    nx_name: Union[str, None] = None,
) -> list[list[str]]:
    """
    Classify a string by keywords + keyword map.
    Uses lemmatization.
    Output indexed by input

    Args:
        strings (list[str]): strings to classify
        keyword_map (Mapping[str, list[str]]): mapping of categories to keywords
        nx_name (str): name to use for non-matches

    Returns: a list of categories

    Example:
        >>> classify_by_keywords(pl.Series(["Method of treatment by tryptamine alkaloids"]), get_patent_attribute_map())
    """
    lookup = create_lookup_map(keyword_map)

    return [classify_string(s, lookup_map=lookup, nx_name=nx_name) for s in strings]
