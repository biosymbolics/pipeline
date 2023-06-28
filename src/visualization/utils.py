"""
Utility functions for visualization.
"""
from typing import NamedTuple
from sklearn import pipeline
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
import polars as pl
import logging
from scipy.sparse import spmatrix  # type: ignore
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.base import TransformerMixin

from clients.spacy import Spacy
from common.utils.dataframe import find_text_columns

MAX_FEATURES = 10000
RANDOM_STATE = 42
MAX_DOC_FREQ = 60
MIN_DOC_FREQ = 2

VectorizationObjects = NamedTuple(
    "VectorizationObjects",
    [
        ("vectorized_data", spmatrix),
        ("feature_names", np.ndarray),
    ],
)


class SpacyLemmatizer(TransformerMixin):
    def __init__(self):
        self.nlp = Spacy.get_instance("en_core_web_sm", disable=["ner"])
        self.nlp.add_pipe("merge_noun_chunks")

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return [self.lemmatize(doc) for doc in X]

    def lemmatize(self, doc):
        return " ".join([token.lemma_ for token in self.nlp(doc)])

    def get_feature_names_out(self, input_features=None):
        return input_features


lemmatizer = SpacyLemmatizer()


def vectorize_data(df: pl.DataFrame, n_features=MAX_FEATURES) -> VectorizationObjects:
    """
    Process the DataFrame to prepare it for dimensionality reduction.
    - Extract all strings and string arrays from the DataFrame (that's it, right now!!)
    - Lemmatize with SpaCy

    Usage:
    ``` python
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Bob', 'David', 'Eve'],
            'age': [25, 32, 41, 28, 36],
            'description': ['A nice PA', 'An accountant', 'A stay at home father', 'Lives in Tokyo', 'Ate the fruit'],
            'education': ['Has a PhD from Harvard', 'finished high school', 'some college', 'dropped out of grade school', 'no education'],
            'score': [8.5, 7.2, 6.8, 9.1, 8.0],
            'diseases': [["asthma", "COPD"], ["PAH"], ["ALZ"], ["PD", "hypertension"], ["asthma"]],
        }
        df = pl.DataFrame(data)
        preprocess_with_tfidf(df)
    ```
    """
    text_columns = find_text_columns(df)
    text_df = df.select(pl.col(text_columns))

    logging.info("Some rows: %s", text_df[0:8])

    if len(text_columns) == 0:
        raise ValueError("No text columns found in the DataFrame")

    # Use ColumnTransformer to apply these different preprocessing steps
    vectorizer = ColumnTransformer(
        transformers=[
            (
                f"{col}_vect",
                Pipeline(
                    [
                        ("lemmatizer", lemmatizer),
                        (
                            "vectorizer",
                            TfidfVectorizer(
                                max_df=MAX_DOC_FREQ,
                                min_df=MIN_DOC_FREQ,
                                max_features=n_features,
                                stop_words="english",
                            ),
                        ),
                    ]
                ),
                col,
            )
            for col in text_columns
        ]
    )
    logging.info("Fitting vectorizer...")

    pipe = pipeline.Pipeline(
        [
            ("transform", vectorizer),
        ]
    )
    data = pipe.fit_transform(text_df.to_pandas())

    return VectorizationObjects(
        vectorized_data=data, feature_names=pipe.get_feature_names_out()
    )
