from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import polars as pl
import logging

from common.utils.dataframe import find_string_array_columns, find_string_columns


def prep_data_for_umap(df: pl.DataFrame):
    """
    Process the DataFrame to prepare it for dimensionality reduction.

    Usage:
    ``` python
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 32, 41, 28, 36],
            'title': ['patent title', 'patent about asthma', 'another one', 'Tokyo patent', 'Sydney patent'],
            'score': [8.5, 7.2, 6.8, 9.1, 8.0],
            'diseases': [["asthma", "COPD"], ["PAH"], ["ALZ"], ["PD", "hypertension"], ["asthma"]],
        }
        df = pl.DataFrame(data)
        prep_data_for_umap(df)
    ```
    """
    vectorizer = TfidfVectorizer()

    def prep_string_array(column):
        series = df.select(pl.col(column).apply(lambda x: " ".join(x))).to_series()
        text_features = vectorizer.fit_transform(series).asformat("array")
        return text_features

    def prep_string(column):
        series = df.select(pl.col(column)).to_series()
        text_features = vectorizer.fit_transform(series).asformat("array")
        return text_features

    def __extract_features(series):
        """
        Flatten the features array into a single array
        """
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(series)
        return scaled_features

    # Preprocess the DataFrame based on column types
    prepped = dict(
        [
            *[
                (column, prep_string_array(column))
                for column in find_string_array_columns(df)
            ],
            *[(column, prep_string(column)) for column in find_string_columns(df)],
        ]
    )

    scaled_features = [
        (column, __extract_features(series)) for column, series in prepped.items()
    ]
    df = pl.DataFrame(dict(scaled_features))
    return df
