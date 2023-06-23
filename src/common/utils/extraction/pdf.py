"""
Extraction tools for PDFs
"""
import logging
import traceback
from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_dict
import polars as pl
import pandas as pd


def get_tables(filename: str = "./PFE-2021.pdf") -> list[pl.DataFrame]:
    """
    Extract tables from PDF and return as dataframes

    Slow due to OCR.

    Args:
        filename (str, optional): path to PDF file. Defaults to "./PFE-2021.pdf".
    """
    try:
        elements = partition(filename=filename, pdf_infer_table_structure=True)
        elem_dict = convert_to_dict(elements)
        tables = [
            pd.read_html(elem["metadata"]["text_as_html"])[0]
            for elem in elem_dict
            if elem["type"] == "Table"
            and elem["metadata"].get("text_as_html") is not None
        ]
    except Exception as e:
        logging.error("Error extracting tables: %s", e)
        raise e
    for table in tables:
        table.head()
    return [pl.from_pandas(table) for table in tables]
