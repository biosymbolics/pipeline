"""
Topic modeling utilities
"""

from typing import Optional, Sequence
from langchain.output_parsers import ResponseSchema
import numpy as np
import polars as pl
import logging

from clients.openai.gpt_client import GptApiClient

RANDOM_STATE = 42

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Topics:
    @staticmethod
    async def generate_descriptions(
        topic_features: pl.DataFrame,
        context_strings: Optional[Sequence[str]] = None,
        existing_labels: Sequence[str] = [],
    ) -> tuple[dict[int, str], dict[int, str]]:
        """
        Ask GPT to guess at good topic labels given a matrix of topic features

        Args:
            topic_features: a dictionary of topic id to list of features
            context_strings: a list of context strings (terms or phrases)
            existing_labels: a list of existing labels

        Returns: a dictionary of topic id to label
        """
        response_schemas = [
            ResponseSchema(
                name="cluster_id", description="original cluster id", type="int"
            ),
            ResponseSchema(name="name", description="concept name", type="string"),
            ResponseSchema(
                name="description", description="concept description", type="string"
            ),
        ]

        def _get_topic_prompt():
            topic_map_desc = "\n".join(
                [
                    f"Topic {topic['cluster_id']}: {', '.join(topic['documents'])}"
                    for topic in topic_features.to_dicts()
                ]
            )
            context_query = (
                "This is the overarching 'concept' under which thse 'sub-concepts' should fall:\n"
                + "\n".join(context_strings)
                if context_strings
                else ""
            )
            existing_label_str = "\n".join(existing_labels)
            exiting_label_query = f"""
                Existing sub-concepts include:
                {existing_label_str}

                Do not duplicate these sub-concepts.
            """
            query = f"""
                Return a succinct name (2-5 words; aka label) and description (2-4 sentences)
                for each sub-concept below.

                {exiting_label_query if len(existing_labels) > 0 else ""}

                Maximize the distinctiveness, specificity and categorical similarity of each sub-concept,
                relative to one another and the existing sub-concepts.
                Better to skip labeling a sub-concept than to provide a bad name.

                Example:
                    name: Galectin-1 antibodies
                    description: "Monovalent antibodies such as nanobodies that are specific for galectin-1 are described.
                    These monovalent antibodies are able to interfere with the activity of galectin-1,
                    and thus may be used for the treatment of diseases associated with dysregulated galectin-1 expression
                    and/or activity, such as certain types of cancers, and conditions associated with pathological angiogenesis."

                {context_query}

                Return a list of json objects (name, description).

                Sub-Concepts to label:
                {topic_map_desc}
            """
            logger.info("Label description prompt: %s", query)
            return query

        client = GptApiClient(response_schemas)
        query = _get_topic_prompt()
        results = await client.query(query, is_array=True)
        print(results)

        topic_name_map: dict[int, str] = dict(
            [(result["cluster_id"], result["name"]) for result in results]
        )
        topic_description_map: dict[int, str] = dict(
            [(result["cluster_id"], result["description"]) for result in results]
        )
        return topic_name_map, topic_description_map

    @staticmethod
    async def model_topics(
        vectorized_docs: Sequence[Sequence[float]],
        documents: Sequence[str],
        existing_labels: Sequence[str] = [],
        context_strings: Optional[Sequence[str]] = None,
    ) -> list[dict]:
        """
        Get topics based on NMF

        Args:
            vectorized_docs (list of list of floats): vectorized data
            documents (list): list of documents
        """
        # lazy loading
        from sklearn.cluster._hdbscan.hdbscan import HDBSCAN

        vectors = np.array(vectorized_docs)

        cluster_ids: list[int] = (
            HDBSCAN(
                # merge clusters that are close
                # https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-cluster-selection-epsilon
                cluster_selection_epsilon=0.4,
                # https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#leaf-clustering
                cluster_selection_method="leaf",
                min_cluster_size=3,
            )
            .fit_predict(vectors)
            .tolist()
        )

        logger.info("Creating topic map (%s clusters)", len(cluster_ids))

        df = (
            pl.DataFrame({"cluster_id": cluster_ids, "documents": documents})
            .group_by("cluster_id")
            .agg(
                [
                    pl.concat_list(pl.col("documents").str.split(" "))
                    .flatten()
                    .unique()
                    .count()
                    .alias("distinct_term_count"),
                    pl.col("documents"),
                ]
            )
        ).filter(pl.col("cluster_id") != -1)

        name_map, description_map = await Topics.generate_descriptions(
            df,
            context_strings=context_strings,
            existing_labels=existing_labels,
        )
        print(name_map, description_map)
        df = df.filter(pl.col("cluster_id").is_in(name_map.keys())).with_columns(
            [
                pl.col("cluster_id").map_dict(name_map).alias("name"),
                pl.col("cluster_id").map_dict(description_map).alias("description"),
            ]
        )

        print(df)

        if len(df) < len(cluster_ids):
            logger.warning(
                "Not all clusters were labeled. Omitting: %s",
                set(cluster_ids) - set(name_map.keys()),
            )

        return df.to_dicts()
