"""
This script applies NER to the patent dataset and saves the results to a temporary location.
"""
from typing import Optional
import polars as pl

from system import initialize

initialize()

from clients.low_level.big_query import execute_bg_query, BQ_DATASET_ID
from common.ner.ner import NerTagger

ID_FIELD = "publication_number"
CHUNK_SIZE = 1000


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


tagger = NerTagger.get_instance(use_llm=True)

# Execute the query
rows = get_rows(last_id=None)
last_id = max(row["publication_number"] for row in rows)

while rows:
    # Convert rows to polars DataFrame
    df = pl.DataFrame([dict(row) for row in rows])

    # Split DataFrame into chunks
    chunks = [df.slice(i, CHUNK_SIZE) for i in range(0, df.shape[0], CHUNK_SIZE)]

    # Process chunks
    for i, chunk in enumerate(chunks):
        # Apply NER
        chunk = chunk.with_columns(
            pl.concat_list(["title", "abstract"])
            .apply(
                lambda x: [(ent.text, ent.label_) for ent in tagger(x).ents],
                return_dtype=pl.Object,
            )
            .alias("entities")
        )

        # Save chunk to a temporary location
        chunk.write_csv(f"chunk_{i}.csv")

    rows = get_rows(last_id=last_id)
    if rows:
        last_id = max(row[ID_FIELD] for row in rows)
