"""
Model-based NER
"""
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline, Pipeline
from pydash import compact

from common.utils.list import dedup

from .types import is_ner_result

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
        model_id (str): model to use
    """
    nlp = __get_pipeline(model_id)
    extracted_batch = nlp(text_batch)

    print(extracted_batch)

    if not isinstance(extracted_batch, list):
        raise Exception("Could not parse response")

    entities = [
        entity.get("word") if is_ner_result(entity) else None
        for entity in extracted_batch
    ]

    return compact(entities)


def extract_named_entities(content: list[str]):
    """
    Extract named entities from a list of content

    Args:
        content (list[str]): list of content on which to do NER
    """
    all_entities = [dedup(__ner_from_batch(batch)) for batch in content]
    return all_entities
