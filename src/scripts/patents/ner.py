"""
This script applies NER to the patent dataset and saves the results to a temporary location.
"""
import asyncio
import json
import logging
import sys
from typing import Callable, Coroutine, Optional
import polars as pl
from common.utils.async_utils import execute_async
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import multiprocessing

from system import initialize

initialize()

from clients.low_level.big_query import execute_bg_query, BQ_DATASET_ID
from common.ner.ner import NerTagger

ID_FIELD = "publication_number"
CHUNK_SIZE = 500


def get_rows(last_id: Optional[str]):
    """
    Get rows from the patent dataset

    Args:
        last_id (str): last id from previous query
    """
    where = f"WHERE {ID_FIELD} > {last_id}" if last_id else ""
    sql = f"""
    SELECT {ID_FIELD}, title, abstract
    FROM `{BQ_DATASET_ID}.gpr_publications`
    {where}
    ORDER BY {ID_FIELD} ASC
    LIMIT 5000
    """
    rows = execute_bg_query(sql)
    return list(rows)


def process_chunk(chunk, i, last_id) -> None:
    logging.info("Processing chunk %s-%s", i, last_id)
    tagger = NerTagger.get_instance(use_llm=True)
    try:
        chunk_ner = chunk.with_columns(
            pl.concat_list(["title", "abstract"])
            .apply(
                lambda x: json.dumps([ent for ent in tagger.extract(x.to_list())]),
            )
            .alias("entities")
        )
        chunk_ner.write_parquet(f"data/ner_output/chunk_{last_id}_{i}.parquet")
    except Exception as e:
        logging.error("Error processing chunk: %s", e)


def generate_ner():
    # Execute the query
    rows = get_rows(last_id=None)
    last_id = max(row["publication_number"] for row in rows)

    while rows:
        # Convert rows to polars DataFrame
        df = pl.DataFrame([dict(row) for row in rows])

        # Split DataFrame into chunks
        chunks = [df.slice(i, CHUNK_SIZE) for i in range(0, df.shape[0], CHUNK_SIZE)]

        # Process chunks
        with ProcessPoolExecutor(max_workers=8) as executor:
            futures = [
                executor.submit(process_chunk, chunk, i, last_id)
                for i, chunk in enumerate(chunks)
            ]
            for future in as_completed(futures):
                future.result()

        rows = get_rows(last_id=last_id)
        if rows:
            last_id = max(row[ID_FIELD] for row in rows)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 ner.py\nLoads NER data for patents and saves it to a temporary location"
        )
        sys.exit()
    generate_ner()
