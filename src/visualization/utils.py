"""
Utility functions for visualization.
"""
from typing import NamedTuple
from sklearn.feature_extraction.text import TfidfVectorizer
import polars as pl
import logging
from scipy.sparse import spmatrix  # type: ignore
import spacy

from common.utils.dataframe import find_string_columns

MAX_FEATURES = 10000
RANDOM_STATE = 42

TfidfObjects = NamedTuple(
    "TfidfObjects",
    [
        ("vectorized_data", spmatrix),
        ("vectorizer", TfidfVectorizer),
    ],
)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])


def __keep_token(token) -> bool:
    """
    -PRON- is special lemma for pronouns
    """
    return token.lemma_ != "-PRON-" and token.pos_ in {
        "NOUN",
        "PROPN",
        "VERB",
        "ADJ",
        "ADV",
    }


MAX_DOC_FREQ = 25
MIN_DOC_FREQ = 2
MAX_NGRAM_LENGTH = 3


def preprocess_with_tfidf(df: pl.DataFrame, n_features=MAX_FEATURES) -> TfidfObjects:
    """
    Process the DataFrame to prepare it for dimensionality reduction.
    - Extract all strings and string arrays from the DataFrame (that's it, right now!!)
    - Lemmatize with SpaCy

    TODO: try DictVectorizer

    Usage:
    ``` python
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 32, 41, 28, 36],
            'description': ['A nice PA', 'An accountant', 'A stay at home father', 'Lives in Tokyo', 'Ate the fruit'],
            'education': ['Has a PhD from Harvard', 'finished high school', 'some college', 'dropped out of grade school', 'no education'],
            'score': [8.5, 7.2, 6.8, 9.1, 8.0],
            'diseases': [["asthma", "COPD"], ["PAH"], ["ALZ"], ["PD", "hypertension"], ["asthma"]],
        }
        df = pl.DataFrame(data)
        preprocess_with_tfidf(df)
    ```
    """
    all_strings: list[list[str]] = (
        df.select(pl.concat_list(find_string_columns(df))).to_series().to_list()
    )

    docs = [nlp(" ".join(s)) for s in all_strings]
    content = [
        " ".join([token.lemma_ for token in doc if __keep_token(token)]) for doc in docs
    ]

    logging.info("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=MAX_DOC_FREQ,
        min_df=MIN_DOC_FREQ,
        max_features=n_features,
        ngram_range=(1, MAX_NGRAM_LENGTH),
        stop_words="english",
    )
    tfidf = tfidf_vectorizer.fit_transform(content)

    return TfidfObjects(
        vectorized_data=tfidf,
        vectorizer=tfidf_vectorizer,
    )
