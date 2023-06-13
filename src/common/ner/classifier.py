import spacy
from typing import Mapping

# 'tagger',
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])


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
    return lookup_map


def classify_by_keywords(
    string: str,
    keyword_map: Mapping[str, list[str]],
    nx_name="OTHER",
) -> list[str]:
    """
    Classify a string by keywords + keyword map.
    Uses lemmatization.

    Args:
        string (str): string to classify
        keyword_map (Mapping[str, list[str]]): mapping of categories to keywords
        nx_name (str): name to use for non-matches

    Returns: a list of categories

    TODO: classify many strings at once
    """
    lookup = create_lookup_map(keyword_map)
    lemmatized_strings = nlp(string).ents
    categories = [lookup.get(doc.lemma_, nx_name) for doc in lemmatized_strings]

    return categories
