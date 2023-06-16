"""
Utility functions for visualization.
"""
from typing import Any, NamedTuple
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
import polars as pl
import logging
from scipy.sparse import spmatrix  # type: ignore
import numpy as np
import spacy

from common.utils.dataframe import find_string_columns

MAX_FEATURES = 10000
RANDOM_STATE = 42

TfidfObjects = NamedTuple(
    "TfidfObjects",
    [
        ("corpus", Any),
        ("tfidf", spmatrix),
        ("tfidf_vectorizer", TfidfVectorizer),
        ("dictionary", corpora.Dictionary),
    ],
)

nlp = spacy.load("en_core_web_sm")


def __keep_token(token) -> bool:
    """
    -PRON- is special lemma for pronouns
    """
    return token.lemma_ != "-PRON-" and token.pos_ in {"NOUN", "VERB", "ADJ", "ADV"}


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
            'title': ['patent title', 'patent about asthma', 'another one', 'Tokyo patent', 'Sydney patent'],
            'score': [8.5, 7.2, 6.8, 9.1, 8.0],
            'diseases': [["asthma", "COPD"], ["PAH"], ["ALZ"], ["PD", "hypertension"], ["asthma"]],
        }
        df = pl.DataFrame(data)
        preprocess_with_tfidf(df)
    ```
    """
    all_strings: list[str] = (
        df.select(pl.concat_list(find_string_columns(df)).flatten())
        .to_series()
        .to_list()
    )

    docs = [nlp(s) for s in all_strings]
    content = [
        " ".join([token.lemma_ for token in doc if __keep_token(token)]) for doc in docs
    ]
    logging.info(content)
    split_content = [doc.split(" ") for doc in content]

    logging.info("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=50,
        min_df=3,
        max_features=n_features,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf = tfidf_vectorizer.fit_transform(content)  # join content?
    dictionary = corpora.Dictionary(split_content, prune_at=20000)
    corpus = [dictionary.doc2bow(text) for text in split_content]

    return TfidfObjects(
        corpus=corpus,
        tfidf=tfidf,
        tfidf_vectorizer=tfidf_vectorizer,
        dictionary=dictionary,
    )
