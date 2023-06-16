"""
Utility functions for visualization.
"""
from typing import Any, NamedTuple
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim import corpora
import polars as pl
import logging
from scipy.sparse import spmatrix  # type: ignore
from sklearn.decomposition import NMF
import numpy as np
import umap

from common.utils.dataframe import find_string_columns
from common.topic import describe_topics

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

NmfObjects = NamedTuple(
    "NmfObjects",
    [("topics", list[str]), ("nmf_embedding", np.ndarray), ("nmf", NMF)],
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


def get_topics(
    tfidf: spmatrix, tfidf_vectorizer: TfidfVectorizer, n_topics: int
) -> NmfObjects:
    """
    Get topics based on NMF

    Feed tfidf stuff from preprocess_with_tfidf()

    Args:
        tfidf: tfidf matrix
        tfidf_vectorizer: tfidf vectorizer
        n_topics: number of topics
    """

    logging.info("Fitting the NMF model with tf-idf features")
    nmf = NMF(n_components=n_topics, random_state=RANDOM_STATE, l1_ratio=0.5).fit(tfidf)

    nmf_embedding = nmf.transform(tfidf)
    feature_names = list(tfidf_vectorizer.get_feature_names_out())

    N_TOP_WORDS = 15

    def __get_feature_names(feature_set: np.ndarray) -> list[str]:
        top_features = feature_set.argsort()[: -N_TOP_WORDS - 1 : -1]
        return [feature_names[i] for i in top_features]

    topic_map = dict(
        [(idx, __get_feature_names(topic)) for idx, topic in enumerate(nmf.components_)]
    )
    topic_name_map = describe_topics(topic_map)

    return NmfObjects(
        topics=list(topic_name_map.values()), nmf_embedding=nmf_embedding, nmf=nmf
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
