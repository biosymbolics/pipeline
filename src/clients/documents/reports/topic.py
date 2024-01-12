from typing import Sequence
import numpy as np

from core.topic import Topics
from typings import ScoredPatent, PatentsTopicReport


def model_patent_topics(
    patents: Sequence[ScoredPatent],
) -> list[PatentsTopicReport]:
    """
    Model patent topics
    """
    # p.embeddings
    embeds = np.array([[] for p in patents])
    names = [p.id for p in patents]
    topics = Topics.model_topics(embeds, names)

    return topics
