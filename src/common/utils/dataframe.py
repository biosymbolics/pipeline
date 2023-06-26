"""
Utility functions for working with DataFrames.
"""
import logging
import polars as pl

THRESHOLD = 0.5


def find_string_array_columns(df: pl.DataFrame, allow_empty: bool = True) -> list[str]:
    """
    Extracts columns that are string arrays

    Args:
        df (pl.DataFrame): DataFrame
        allow_empty (bool, optional): Allow empty string arrays. Defaults to True.
    """
    string_array_columns = [
        column
        for column in df.columns
        if df.select(pl.col(column))
        .to_series()
        .apply(
            lambda x: isinstance(x, pl.Series)
            and (allow_empty or (len(x) > 0 and isinstance(x[0], str)))
        )
        .all()
    ]
    return string_array_columns


def find_string_columns(df: pl.DataFrame) -> list[str]:
    """
    Extracts columns that are strings

    Args:
        df (pl.DataFrame): DataFrame
    """
    string_columns = [
        column
        for column in df.columns
        if df[column].apply(lambda x: isinstance(x, str)).all()
    ]
    return string_columns


def find_text_columns(df: pl.DataFrame) -> list[str]:
    """
    Extracts columns that are text (long strings)

    Args:
        df (pl.DataFrame): DataFrame
    """
    string_columns = [
        column
        for column in df.columns
        if (
            df[column]
            .apply(
                lambda x: isinstance(x, str) and len(x) > 35 and len(x.split(" ")) > 3
            )
            .mean()
            or 0 > THRESHOLD
        )
    ]
    return string_columns
