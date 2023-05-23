"""
Model-based NER
"""
from typing import TypedDict
import torch
from tqdm.auto import tqdm
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline, Pipeline
from pydash import flatten

from common.utils.list import dedup

NerResult = TypedDict("NerResult", {"word": str, "score": float, "entity": str})


# set device to GPU if available
DEVICE = torch.cuda.current_device() if torch.cuda.is_available() else None


DEFAULT_MODEL_ID = "dslim/bert-base-NER"


def __get_pipeline(model_id: str = DEFAULT_MODEL_ID) -> Pipeline:
    """
    Initialize and return NER pipeline
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)
    ner_pipeline = pipeline(
        "ner",
        model=model,
        tokenizer=tokenizer,
        aggregation_strategy="max",
        device=DEVICE,
    )

    return ner_pipeline


def __ner_from_batch(text_batch: str, model_id: str = DEFAULT_MODEL_ID) -> list[str]:
    """
    Extract named entities from a batch of text

    Args:
        text_batch (str): text on which to do NER
    """
    nlp = __get_pipeline()
    extracted_batch = nlp(text_batch, model_id)

    if not isinstance(extracted_batch, list):
        raise Exception("Could not parse response")

    entities: list[list[str]] = []

    for result in extracted_batch:
        if isinstance(result, list):
            ne = [entity.word for entity in result]
            entities.append(ne)

        raise Exception("Could not parse response")

    return flatten(entities)


def extract_named_entities(content: list[str]):
    """
    Extract named entities from a list of content

    Args:
        content (list[str]): list of content on which to do NER
    """
    all_entities = [dedup(__ner_from_batch(batch)) for batch in content]
    return all_entities
