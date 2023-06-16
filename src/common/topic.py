"""
Topic modeling utilities
"""
from typing import Optional, NamedTuple
from langchain.output_parsers import ResponseSchema
from scipy.sparse import spmatrix  # type: ignore
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import umap
import numpy as np
import polars as pl
import logging

from clients.openai.gpt_client import GptApiClient
from common.utils.dataframe import find_string_columns

RANDOM_STATE = 42
KNN = 5

TopicObjects = NamedTuple(
    "TopicObjects",
    [("topics", list[str]), ("topic_embedding", np.ndarray)],
)


def describe_topics(
    topic_features: dict[int, list[str]], context_terms: Optional[list[str]] = None
) -> dict[int, str]:
    """
    Ask GPT to guess at good topic labels given a matrix of topic features

    Args:
        topic_features: a dictionary of topic id to list of features

    Returns: a dictionary of topic id to label
    """
    response_schemas = [
        ResponseSchema(name="id", description="the original topic id (int)"),
        ResponseSchema(name="label", description="the label (str)"),
        # ResponseSchema(
        #     name="description",
        #     description="a detailed, technical description of what documents this topic contains (str)",
        # ),
    ]

    client = GptApiClient(response_schemas)
    topic_map_desc = [
        f"Topic {idx}: {', '.join(features)}"
        for idx, features in topic_features.items()
    ]
    context_query = (
        " given the context of " + ", ".join(context_terms) if context_terms else ""
    )
    query = f"""
        Return a descriptive, succinct name (4 words or fewer) for each topic below{context_query},
        maximizing orthagonality:
        {topic_map_desc}
    """

    results = client.query(query, is_array=True)

    if not isinstance(results, list):
        logging.error(results)
        raise ValueError(f"Expected list of results, got {type(results)}")

    topic_map = dict([(result["id"], result["label"]) for result in results])
    return topic_map


def get_topics(
    fitted_matrix: spmatrix,
    feature_names: list[str],
    n_topics: int,
    n_top_words: int,
    context_terms: Optional[list[str]] = None,
) -> TopicObjects:
    """
    Get topics based on NMF

    Args:
        fitted_matrix: fitted matrix
        feature_names: vectorizer
        n_topics: number of topics
        n_top_words: number of top words to use in description
    """

    logging.info("Fitting the NMF model with tf-idf features")
    nmf = NMF(n_components=n_topics, random_state=RANDOM_STATE, l1_ratio=0.5).fit(
        fitted_matrix
    )
    nmf_embedding = nmf.transform(fitted_matrix)

    def __get_feature_names(feature_set: np.ndarray) -> list[str]:
        top_features = feature_set.argsort()[: -n_top_words - 1 : -1]
        return [feature_names[i] for i in top_features]

    topic_map = dict(
        [(idx, __get_feature_names(topic)) for idx, topic in enumerate(nmf.components_)]
    )
    topic_name_map = describe_topics(topic_map, context_terms)

    return TopicObjects(
        topics=list(topic_name_map.values()), topic_embedding=nmf_embedding
    )


def calculate_umap_embedding(
    tfidf: spmatrix, knn: int = KNN, min_dist: float = 0.001
) -> pl.DataFrame:
    """
    Calculate the UMAP embedding

    Args:
        tfidf: tfidf matrix
        knn: number of nearest neighbors
        min_dist: minimum distance

    Returns: UMAP embedding in a DataFrame (x, y)
    """
    logging.info("Attempting UMAP")
    umap_embr = umap.UMAP(
        n_neighbors=knn, metric="cosine", min_dist=min_dist, random_state=RANDOM_STATE
    )
    embedding = umap_embr.fit_transform(tfidf.toarray())

    if not isinstance(embedding, np.ndarray):
        raise TypeError("UMAP embedding is not a numpy array")
    embedding = pl.from_numpy(embedding, schema={"x": pl.Float32, "y": pl.Int64})
    return embedding


def get_topics_with_bert(df: pl.DataFrame):
    """
    Get topics based on BERTopic

    Args:
        df: dataframe
    """
    vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")

    logging.info("BERT")
    all_strings: list[list[str]] = (
        df.select(pl.concat_list(find_string_columns(df))).to_series().to_list()
    )
    docs = [" ".join(s) for s in all_strings]
    model = BERTopic(
        vectorizer_model=vectorizer_model,
        language="english",
        calculate_probabilities=True,
        verbose=True,
    )
    topics = model.fit_transform(docs)
    freq = model.get_topic_info()
    logging.info("Frequency of topics: %s", freq)
    return topics
