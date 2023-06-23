"""
Extraction tools for PDFs
"""
from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_dict
import polars as pl
import pandas as pd


def get_tables(filename: str = "./PFE-2021.pdf"):
    """
    Extract tables from PDF and return as dataframes

    Slow due to OCR.

    Args:
        filename (str, optional): path to PDF file. Defaults to "./PFE-2021.pdf".
    """
    elements = partition(filename=filename, pdf_infer_table_structure=True)
    elem_dict = convert_to_dict(elements)
    tables = [
        pl.from_dataframe(pd.read_html(elem["text_as_html"]))
        for elem in elem_dict
        if elem["type"] == "Table"
    ]
    for table in tables:
        table.head()
    return tables
