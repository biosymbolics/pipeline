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
import umap

from common.utils.dataframe import find_string_columns

MAX_FEATURES = 10000
KNN = 5
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


def preprocess_with_tfidf(df: pl.DataFrame, n_features=MAX_FEATURES) -> TfidfObjects:
    """
    Process the DataFrame to prepare it for dimensionality reduction.
    - Extract all strings and string arrays from the DataFrame (that's it, right now!!)

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
    # all_tags: list[str] = (
    #     df.select(pl.concat_list(find_string_array_columns(df)).flatten())
    #     .to_series()
    #     .to_list()
    # )
    content = [*all_strings]  # , *all_tags]
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


def caculate_umap_embedding(tfidf: spmatrix) -> pl.DataFrame:
    """
    Calculate the UMAP embedding

    Args:
        tfidf: tfidf matrix

    Returns: UMAP embedding in a DataFrame (x, y)
    """
    logging.info("Attempting UMAP")
    umap_embr = umap.UMAP(
        n_neighbors=KNN, metric="cosine", min_dist=0.1, random_state=RANDOM_STATE
    )
    embedding = umap_embr.fit_transform(tfidf.toarray())

    if not isinstance(embedding, np.ndarray):
        raise TypeError("UMAP embedding is not a numpy array")
    embedding = pl.from_numpy(embedding, schema={"x": pl.Float32, "y": pl.Int64})
    return embedding
