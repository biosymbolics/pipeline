from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
import polars as pl
import logging


def prep_data_for_umap(df: pl.DataFrame):
    """
    Process the DataFrame to prepare it for dimensionality reduction.

    Usage:
    ``` python
        data = {
            'id': [1, 2, 3, 4, 5],
            'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
            'age': [25, 32, 41, 28, 36],
            'city': ['New York', 'Paris', 'London', 'Tokyo', 'Sydney'],
            'score': [8.5, 7.2, 6.8, 9.1, 8.0],
            'diseases': [["asthma", "COPD"], ["PAH"], ["ALZ"], ["PD", "hypertension"], ["asthma"]],
        }
        df = pl.DataFrame(data)
        prep_data_for_umap(df)
    ```
    """
    vectorizer = TfidfVectorizer()

    def preprocess_string_array(column):
        series = df.select(pl.col(column).apply(lambda x: " ".join(x))).to_series()
        text_features = vectorizer.fit_transform(series).asformat("array")
        return text_features

    def preprocess_string(column):
        series = df.select(pl.col(column)).to_series()
        text_features = vectorizer.fit_transform(series).asformat("array")
        return text_features

    def __flatten_features(series):
        """
        Flatten the features array into a single array
        """
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(series)
        return scaled_features

    column_types = {
        "string_array": [
            column
            for column in df.columns
            if df[column]
            .apply(
                lambda x: isinstance(x, pl.Series)
                and len(x) > 0
                and isinstance(x[0], str)
            )
            .all()
        ],
        "string": [
            column
            for column in df.columns
            if df[column].apply(lambda x: isinstance(x, str)).all()
        ],
    }

    logging.info("Column types: %s", column_types)

    # Preprocess the DataFrame based on column types
    preprocessed = dict(
        [
            *[
                (column, preprocess_string_array(column))
                for column in column_types["string_array"]
            ],
            *[(column, preprocess_string(column)) for column in column_types["string"]],
        ]
    )

    scaled_features = [
        (column, __flatten_features(series)) for column, series in preprocessed.items()
    ]
    df = pl.DataFrame(dict(scaled_features))
    return df
