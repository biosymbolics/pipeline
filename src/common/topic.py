"""
Topic modeling utilities
"""
from langchain.output_parsers import ResponseSchema
import logging

from clients.openai.gpt_client import GptApiClient


def describe_topics(topic_features: dict[int, list[str]]) -> dict[int, str]:
    """
    Ask GPT to guess at good topic labels given a matrix of topic features

    Args:
        topic_features: a dictionary of topic id to list of features

    Returns: a dictionary of topic id to label
    """
    response_schemas = [
        ResponseSchema(name="id", description="the original topic id (int)"),
        ResponseSchema(name="label", description="the label (str)"),
    ]

    client = GptApiClient(response_schemas)
    topic_map_desc = [
        f"Topic {idx}: {', '.join(features)}"
        for idx, features in topic_features.items()
    ]
    query = f"""
        Return a good, succinct name (4 words or fewer) for each topic below, maximizing the contrast between topics:
        {topic_map_desc}
    """

    results = client.query(query, is_array=True)

    if not isinstance(results, list):
        logging.error(results)
        raise ValueError(f"Expected list of results, got {type(results)}")

    topic_map = dict([(result["id"], result["label"]) for result in results])
    return topic_map
