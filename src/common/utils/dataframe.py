"""
Utility functions for working with DataFrames.
"""
import polars as pl


def find_string_array_columns(df: pl.DataFrame) -> list[str]:
    """
    Extracts columns that are string arrays

    Args:
        df (pl.DataFrame): DataFrame
    """
    string_array_columns = [
        column
        for column in df.columns
        if df[column]
        .apply(
            lambda x: isinstance(x, pl.Series) and len(x) > 0 and isinstance(x[0], str)
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
