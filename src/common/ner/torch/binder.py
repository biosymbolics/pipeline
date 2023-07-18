import copy
from functools import reduce
from typing import Any
import numpy as np
from pydash import flatten
import torch
from transformers import AutoTokenizer
import logging
import polars as pl

from scripts.patents.binder import generate_word_indices

logger = logging.getLogger(__name__)

base_model = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
tokenizer = AutoTokenizer.from_pretrained(base_model)

DOC_STRIDE = 16
MAX_LENGTH = 512


class Annotation:
    id: str
    entity_type: str
    start_char: int
    end_char: int
    text: str

    def __init__(self, id, entity_type, start_char, end_char, text):
        self.id = id
        self.entity_type = entity_type
        self.start_char = start_char
        self.end_char = end_char
        self.text = text


def prepare_features(example, split: str = "dev"):
    tokenized_example: Any = tokenizer(
        example["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        stride=DOC_STRIDE,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="longest",
    )

    sample_mapping = tokenized_example.pop("overflow_to_sample_mapping")

    tokenized_example["text"] = example["text"]
    tokenized_example["split"] = []
    tokenized_example["id"] = example["id"]
    tokenized_example["token_start_mask"] = []
    tokenized_example["token_end_mask"] = []

    for i in range(len(tokenized_example["input_ids"])):
        tokenized_example["split"].append(split)

        # Grab the sequence corresponding to that example (to know what is the text and what are special tokens).
        sequence_ids = tokenized_example.sequence_ids(i)

        # One example can give several texts, this is the index of the example containing this text.
        sample_index = sample_mapping[i]

        # Create token_start_mask and token_end_mask where mask = 1 if the corresponding token is either a start
        # or an end of a word in the original dataset.
        token_start_mask, token_end_mask = [], []
        word_start_chars = example["word_start_chars"][sample_index]
        word_end_chars = example["word_end_chars"][sample_index]
        for index, (start_char, end_char) in enumerate(
            tokenized_example["offset_mapping"][i]
        ):
            if sequence_ids[index] != 0:
                token_start_mask.append(0)
                token_end_mask.append(0)
            else:
                token_start_mask.append(int(start_char in word_start_chars))
                token_end_mask.append(int(end_char in word_end_chars))

        tokenized_example["token_start_mask"].append(token_start_mask)
        tokenized_example["token_end_mask"].append(token_end_mask)

        tokenized_example["offset_mapping"][i] = [
            (o if sequence_ids[k] == 0 else None)
            for k, o in enumerate(tokenized_example["offset_mapping"][i])
        ]

    num_examples = len(tokenized_example["id"])
    features = [
        {key: value[i] for key, value in tokenized_example.items()}
        for i in range(num_examples)
    ]
    return features


def extract_predictions(
    features,
    predictions: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> list:
    if len(predictions) != 3:
        print(predictions)
        raise ValueError(
            "`predictions` should be a tuple with three elements (start_logits, end_logits, span_logits)."
        )
    _, _, all_span_logits = predictions

    if len(predictions[0]) != len(features):
        raise ValueError(
            f"Got {len(predictions[0])} predictions and {len(features)} features."
        )

    def get_prediction(feature, idx):
        # We grab the masks for start and end indices.
        token_start_mask = np.array(feature["token_start_mask"]).astype(bool)
        token_end_mask = np.array(feature["token_end_mask"]).astype(bool)

        # We grab the predictions of the model for this feature.
        span_logits = all_span_logits[idx]

        # We use the [CLS] logits as thresholds
        span_preds = np.triu(span_logits > span_logits[:, 0:1, 0:1])

        type_ids, start_indexes, end_indexes = (
            token_start_mask[np.newaxis, :, np.newaxis]
            & token_end_mask[np.newaxis, np.newaxis, :]
            & span_preds
        ).nonzero()

        # This is what will allow us to map some the positions in our logits to span of texts in the original context.
        offset_mapping = features[idx]["offset_mapping"]

        def create_annotation(rec):
            start_char, end_char = (
                offset_mapping[rec["start_index"]][0],
                offset_mapping[rec["end_index"]][1],
            )
            pred = Annotation(
                id=feature["id"],
                entity_type=rec["type_id"],
                start_char=start_char,
                end_char=end_char,
                text=feature["text"][start_char:end_char],
            )
            return pred

        example_predictions = [
            create_annotation(rec) for rec in zip(type_ids, start_indexes, end_indexes)
        ]

        return example_predictions

    all_predictions = flatten(
        [get_prediction(feature, idx) for idx, feature in enumerate(features)]
    )

    return all_predictions


def prepare_example(text: str) -> dict:
    word_idx = generate_word_indices(text)
    return {
        "text": [text],
        "id": [1],
        "word_start_chars": [[word[0] for word in word_idx]],
        "word_end_chars": [[word[1] for word in word_idx]],
    }


types = [
    {
        "dataset": "BIOSYM",
        "name": "compounds",
        "description": "in this context, compounds are chemical or biological substances, drug classes or drug names",
        "description_source": "",
    },
    {
        "dataset": "BIOSYM",
        "name": "diseases",
        "description": "diseases are conditions and indications for which pharmacological treatments are developed",
        "description_source": "",
    },
    {
        "dataset": "BIOSYM",
        "name": "mechanisms",
        "description": "mechanisms of action are ways in which a drug has an effect (inhibitors, agonists, etc)",
        "description_source": "",
    },
]


def predict(text="this patent is for a novel anti-amyloid monoclonal antibody"):
    model = torch.load("model.pt")

    features = prepare_features(prepare_example(text))
    inputs = tokenizer(text, return_tensors="pt")

    tokenized_descriptions = tokenizer(
        [t["description"] for t in types],
        truncation=True,
        padding="longest",
        return_tensors="pt",
    )

    data_collator = {
        "type_input_ids": tokenized_descriptions["input_ids"],
        "type_attention_mask": tokenized_descriptions["attention_mask"],
        "type_token_type_ids": tokenized_descriptions["token_type_ids"]
        if "token_type_ids" in tokenized_descriptions
        else None,
    }

    predictions = model(**inputs, **data_collator).__dict__
    predictions = (
        predictions["start_scores"],
        predictions["end_scores"],
        predictions["span_scores"],
    )

    results = extract_predictions(features, predictions)
    return results
