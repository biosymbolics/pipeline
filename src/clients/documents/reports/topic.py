from typing import Sequence
import numpy as np

from core.topic import Topics
from typings import PatentApplication, PatentsTopicReport


def model_patent_topics(
    patents: Sequence[PatentApplication],
) -> list[PatentsTopicReport]:
    """
    Model patent topics
    """
    embeds = np.array([p.embeddings for p in patents])
    names = [p.publication_number for p in patents]
    topics = Topics.model_topics(embeds, names)

    return topics
