"""
Text classifiers
"""
from functools import partial
from pydash import compact
import spacy
from typing import Mapping, Union
import logging
import polars as pl

nlp = spacy.load("en_core_web_sm", disable=["ner"])


def create_lookup_map(keyword_map: Mapping[str, list[str]]) -> Mapping[str, str]:
    """
    Create a lookup map from a keyword map, with keywords lemmatized.

    Example:
    ``` python
    keyword_map = {
        "INDICATION": ["indication", "diseases"],
        "INTERVENTION": ["intervention", "drug", "compounds"],
    }
    yields lemmatized keyword to category mapping:
    {
        'indication': 'INDICATION',
        'disease': 'INDICATION',
        'drug': 'INTERVENTION',
        'compound': 'INTERVENTION',
    }
    ```
    """
    cat_key_tups = [
        (keyword, category)
        for category, keywords in keyword_map.items()
        for keyword in keywords
    ]
    lookup_tups = [
        (kw_tok.lemma_, tup[1]) for tup in cat_key_tups for kw_tok in nlp(tup[0])
    ]
    lookup_map = dict(lookup_tups)

    logging.debug(f"Created lookup map: %s", lookup_map)
    return lookup_map


def classify_string(
    string: str, lookup_map: Mapping[str, str], nx_name: Union[str, None] = "OTHER"
) -> list[str]:
    """
    Classify a string by keywords + lookup map.
    Assumes lookup map has lemmatized keywords.

    Args:
        string (str): string to classify
        lookup_map (Mapping[str, str]): mapping of lemmatized keywords to categories
        nx_name (str): name to use for non-matches
    """
    tokens = nlp(string)
    categories = [lookup_map.get(token.lemma_, nx_name) for token in tokens]
    return compact(categories)


def classify_by_keywords(
    strings: pl.Series,
    keyword_map: Mapping[str, list[str]],
    nx_name: Union[str, None] = "OTHER",
) -> pl.Series:
    """
    Classify a string by keywords + keyword map.
    Uses lemmatization.

    Args:
        string (str): string to classify
        keyword_map (Mapping[str, list[str]]): mapping of categories to keywords
        nx_name (str): name to use for non-matches

    Returns: a list of categories
    """
    lookup = create_lookup_map(keyword_map)

    __classify_string = partial(classify_string, lookup_map=lookup, nx_name=nx_name)

    return strings.apply(__classify_string)
