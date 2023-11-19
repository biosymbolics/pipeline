"""
Topic modeling utilities
"""
from typing import Mapping, NamedTuple, Optional, Sequence
from langchain.output_parsers import ResponseSchema
import numpy as np
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import umap
import numpy.typing as npt
import polars as pl
import logging

from clients.openai.gpt_client import GptApiClient
from utils.dataframe import find_string_columns

RANDOM_STATE = 42


class TopicObjects(NamedTuple):
    topics: list[str]
    topic_embedding: npt.NDArray
    dictionary: npt.NDArray  # would like to type this more specifically... i think (int, float) but not sure


class Topics:
    @staticmethod
    def generate_descriptions(
        topic_features: Mapping[int, Sequence[str]],
        context_terms: Optional[Sequence[str]] = None,
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

        def _get_topic_prompt():
            topic_map_desc = [
                f"Topic {idx}: {', '.join(features)}\n"
                for idx, features in topic_features.items()
            ]
            context_query = (
                " given the context of " + ", ".join(context_terms)
                if context_terms
                else ""
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
        query = _get_topic_prompt()
        results = client.query(query, is_array=True)

        if not isinstance(results, list):
            logging.error(results)
            raise ValueError(f"Expected list of results, got {type(results)}")

        topic_map = dict([(result["id"], result["label"]) for result in results])
        return topic_map

    @staticmethod
    def model_topics(
        vectorized_data: npt.NDArray,
        feature_names: npt.NDArray,
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

        logging.info("Fitting NMF now")
        nmf = nmf.fit(vectorized_data)
        logging.info("Transforming NMF now")
        embedding = nmf.transform(vectorized_data)
        dictionary = nmf.components_  # aka factorization matrix

        logging.info("Creating topic map")

        def _get_feat_names(feature_set: npt.NDArray) -> list[str]:
            top_features = feature_set.argsort()[: -n_top_words - 1 : -1]
            return [str(feature_names[i]) for i in top_features]

        topic_map = dict(
            [(idx, _get_feat_names(topic)) for idx, topic in enumerate(dictionary)]
        )
        topic_name_map = Topics.generate_descriptions(
            topic_map, context_terms=context_terms
        )

        return TopicObjects(
            topics=list(topic_name_map.values()),
            topic_embedding=embedding,
            dictionary=dictionary,
        )

    @staticmethod
    def model_topics_with_bert(df: pl.DataFrame) -> npt.NDArray:
        """
        Get topics based on BERTopic
        Gets all string columns from df, concats into a single string,
        fit/transforms with BERTopic, and returns the topics

        Args:
            df: dataframe
        """
        vectorizer_model = CountVectorizer(ngram_range=(1, 2), stop_words="english")

        logging.info("Modeling topics with BERTopic")
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
        topics = np.array(model.fit_transform(docs))
        freq = model.get_topic_info()
        logging.info("Frequency of topics: %s", freq)
        return topics


def calculate_umap_embedding(
    vectorized_data: npt.NDArray,
    dictionary: npt.NDArray,
    knn: int = 10,
    min_dist: float = 0.75,
) -> tuple[pl.DataFrame, npt.NDArray]:
    """
    Calculate the UMAP embedding

    Args:
        vectorized_data: vectorized data (tfidf matrix)
        dictionary (npt.NDArray): factorization matrix
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
    embedding = np.array(umap_embr.fit_transform(vectorized_data))
    embedding_df = pl.from_numpy(embedding, schema={"x": pl.Float32, "y": pl.Float32})
    centroids = np.array(umap_embr.transform(dictionary))

    return embedding_df, centroids
