"""
Extraction tools for PDFs
"""
import logging
import os
import time
import traceback
from unstructured.partition.auto import partition
from unstructured.staging.base import convert_to_dict
import polars as pl

from common.utils.file import save_as_pickle

API_KEY = os.environ["UNSTRUCTURED_API_KEY"]


def get_tables(filename: str) -> list[pl.DataFrame]:
    """
    Extract tables from PDF and return as dataframes

    Slow due to OCR.

    Args:
        filename (str, optional): path to PDF file. Defaults to "./PFE-2021.pdf".

    API:
    curl -X 'POST' \
    'https://api.unstructured.io/general/v0/general' \
    -H 'accept: application/json' \
    -H 'Content-Type: multipart/form-data' \
    -F 'files=@/tmp/sec.pdf' \
    -F 'strategy=hi_res' \
    -F 'pdf_infer_table_structure=true' \
    -H 'unstructured-api-key: API_KEY' \
    | jq -C . | less -R
    """
    raise NotImplementedError("TODO: OCR is too slow")
    # start = time.time()
    # try:
    #     elements = partition(filename=filename, pdf_infer_table_structure=True)
    #     elem_dict = convert_to_dict(elements)
    #     tables = [
    #         # pl.read_html(elem["metadata"]["text_as_html"])[0]
    #         html = ""
    #         for elem in elem_dict
    #         if elem["type"] == "Table"
    #         and elem["metadata"].get("text_as_html") is not None
    #     ]
    # except Exception as e:
    #     logging.error("Error extracting tables: %s", e)
    #     raise e

    # # df_tables = [pl.from_pandas(table) for table in tables]
    # df_tables = tables

    # for df in df_tables:
    #     save_as_pickle(df, "table-df", True)

    # logging.info("Took %s seconds to extract tables", time.time() - start)
    # return df_tables
