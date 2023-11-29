"""
Topic modeling utilities
"""
from typing import Mapping, Optional, Sequence, cast
from langchain.output_parsers import ResponseSchema
from sklearn.decomposition import PCA
import numpy.typing as npt
import polars as pl
import logging

from clients.openai.gpt_client import GptApiClient
from typings.patents import PatentsTopicReport

RANDOM_STATE = 42


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
        feature_names: Sequence[str],
        n_dim: int = 2,
    ) -> list[PatentsTopicReport]:
        """
        Get topics based on NMF

        Args:
            vectorized_data (spmatrix): vectorized data
            n_dim (int): number of dims

        from clients.patents import search_client
        from handlers.patents.types import PatentSearchParams
        import json
        p = search_client.search(PatentSearchParams(terms=['migraine disorders'], skip_cache=True, limit=50))
        from data.topic.topic import Topics
        embeds = [json.loads(t.embeddings) for t in p]
        names = [t.title for t in p]
        Topics.model_topics(embeds, names)
        """

        logging.info("Fitting the NMF model with tf-idf features")
        pca = PCA(n_components=n_dim, random_state=RANDOM_STATE)

        logging.info("Fitting pca now")
        pca = pca.fit(vectorized_data)
        logging.info("Transforming pca now")
        embedding = pca.transform(vectorized_data)

        logging.info("Creating topic map")

        embedding_df = pl.from_numpy(
            embedding,
            schema={"x": pl.Float32, "y": pl.Float32},
        )
        embedding_df = embedding_df.with_columns(
            pl.Series(feature_names).alias("publication_number"),
        )

        return cast(list[PatentsTopicReport], embedding_df.to_dicts())
