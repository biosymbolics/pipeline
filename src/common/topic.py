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
KNN = 10


class TopicObjects(NamedTuple):
    topics: list[str]
    topic_embedding: np.ndarray
    dictionary: np.ndarray  # would like to type this more specifically... i think (int, float) but not sure


def describe_topics(
    topic_features: dict[int, list[str]], context_terms: Optional[list[str]] = None
) -> dict[int, str]:
    """
    Ask GPT to guess at good topic labels given a matrix of topic features

    Args:
        topic_features: a dictionary of topic id to list of features
        context_terms: a list of context terms

    Returns: a dictionary of topic id to label
    """
    response_schemas = [
        ResponseSchema(name="id", description="the original topic id (int)"),
        ResponseSchema(name="label", description="the label (str)"),
    ]

    def __get_topic_prompt():
        topic_map_desc = [
            f"Topic {idx}: {', '.join(features)}\n"
            for idx, features in topic_features.items()
        ]
        context_query = (
            " given the context of " + ", ".join(context_terms) if context_terms else ""
        )
        query = f"""
            Return a descriptive, succinct name (4 words or fewer) for each topic below{context_query},
            maximizing orthagonality and semantic meaningfulness of the labels. Return a list of json objects.

            Topics:
            {topic_map_desc}
        """
        logging.debug("Label description prompt: %s", query)
        return query

    client = GptApiClient(response_schemas)
    query = __get_topic_prompt()
    results = client.query(query, is_array=True)

    if not isinstance(results, list):
        logging.error(results)
        raise ValueError(f"Expected list of results, got {type(results)}")

    topic_map = dict([(result["id"], result["label"]) for result in results])
    return topic_map


def get_topics(
    vectorized_data: spmatrix,
    feature_names: np.ndarray,
    n_topics: int,
    n_top_words: int,
    context_terms: Optional[list[str]] = None,
) -> TopicObjects:
    """
    Get topics based on NMF

    Args:
        vectorized_data (spmatrix): vectorized data
        n_topics (int): number of topics
        n_top_words (int): number of top words to use in description
        context_terms (Optional[list[str]]): context terms
    """

    logging.info("Fitting the NMF model with tf-idf features")
    nmf = NMF(n_components=n_topics, random_state=RANDOM_STATE, l1_ratio=0.5)

    logging.info("Fitting now")
    nmf = nmf.fit(vectorized_data)
    logging.info("Transforming now")
    embedding = nmf.transform(vectorized_data)
    dictionary = nmf.components_  # aka factorization matrix

    logging.info("Creating topic map")

    def __get_feat_names(feature_set: np.ndarray) -> list[str]:
        top_features = feature_set.argsort()[: -n_top_words - 1 : -1]
        logging.info("Top features: %s", top_features)
        return [str(feature_names[i]) for i in top_features]

    topic_map = dict(
        [(idx, __get_feat_names(topic)) for idx, topic in enumerate(dictionary)]
    )
    topic_name_map = describe_topics(topic_map, context_terms=context_terms)

    return TopicObjects(
        topics=list(topic_name_map.values()),
        topic_embedding=embedding,
        dictionary=dictionary,
    )


def calculate_umap_embedding(
    vectorized_data: spmatrix,
    dictionary: np.ndarray,
    knn: int = KNN,
    min_dist: float = 0.2,
) -> tuple[pl.DataFrame, np.ndarray]:
    """
    Calculate the UMAP embedding

    Args:
        vectorized_data: vectorized data (tfidf matrix)
        dictionary (np.ndarray): factorization matrix (aka dictionary)
        knn (int): number of nearest neighbors
        min_dist (float): minimum distance

    Returns: UMAP embedding in a DataFrame (x, y)
    """
    logging.info("Starting UMAP")
    umap_embr = umap.UMAP(
        n_neighbors=knn,
        metric="euclidean",
        min_dist=min_dist,
        random_state=RANDOM_STATE,
    )
    embedding = umap_embr.fit_transform(vectorized_data.toarray())

    if not isinstance(embedding, np.ndarray):
        raise TypeError("UMAP embedding is not a numpy array")

    embedding = pl.from_numpy(embedding, schema={"x": pl.Float32, "y": pl.Int64})

    centroids = umap_embr.transform(dictionary)

    return embedding, centroids


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
