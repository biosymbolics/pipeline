import json
import logging
from typing import List, Optional
import polars as pl
from clients.low_level.big_query import select_from_bg, upsert_into_bg_table

from common.ner.ner import NerTagger
from common.utils.string import get_id

ID_FIELD = "publication_number"
CHUNK_SIZE = 50

MAX_TEXT_LENGTH = 500
DECAY_RATE = 1 / 2000
PROCESSED_PUBS_FILE = "data/processed_pubs.txt"
BASE_DIR = "data/ner_enriched"
MIN_SEARCH_RANK = 0.1


def get_patents(terms: list[str], last_id: Optional[str] = None) -> list[dict]:
    """
    Get patents from BigQuery

    Args:
        terms (list[str]): terms on which to search for patents
        last_id (Optional[str], optional): last id to paginate from. Defaults to None.
    """
    lower_terms = [term.lower() for term in terms]

    pagination_where = f"AND {ID_FIELD} > '{last_id}'" if last_id else ""

    query = f"""
        WITH matches AS (
            SELECT
                a.publication_number as publication_number,
                EXP(-annotation.character_offset_start * {DECAY_RATE}) as search_rank, --- exp decay scaling; higher is better
            FROM patents.annotations a,
            UNNEST(a.annotations) as annotation
            WHERE annotation.term IN UNNEST({lower_terms})
        )
        SELECT apps.publication_number, apps.title, apps.abstract
        FROM patents.applications AS apps, matches
        WHERE apps.publication_number = matches.publication_number
        AND search_rank > {MIN_SEARCH_RANK}
        {pagination_where}
        ORDER BY {ID_FIELD} DESC
        limit {CHUNK_SIZE}
    """
    patents = select_from_bg(query)
    return patents


def get_patent_descriptions(patents: pl.DataFrame) -> list[str]:
    """
    Get patent descriptions (title + abstract)

    - Concatenates title and abstract into `text` column
    - Truncates `text` column to MAX_TEXT_LENGTH
    - Returns list of text (one line per patent)

    Args:
        patents (pl.DataFrame): patents to preprocess
    """
    df = patents.with_columns(
        pl.concat_str(["title", "abstract"], separator="\n").alias("text"),
    ).with_columns(
        pl.when(pl.col("text").str.lengths() >= MAX_TEXT_LENGTH)
        .then(pl.col("text").str.slice(0, MAX_TEXT_LENGTH))
        .otherwise(pl.col("text"))
        .alias("text")
    )

    return df["text"].to_list()


def enrich_patents(patents: pl.DataFrame) -> pl.DataFrame:
    """
    Enriches patents with entities

    Args:
        patents (pl.DataFrame): patents to enrich

    Returns:
        pl.DataFrame: enriched patents
    """

    def format(df: pl.DataFrame):
        """
        Unpacks entities into separate rows
        """
        flattened_df = (
            df.explode("entities")
            .lazy()
            .select(
                pl.col("publication_number"),
                pl.col("entities").apply(lambda e: e[0]).alias("term"),
                pl.col("entities").apply(lambda e: e[1]).alias("domain"),
                pl.lit("title+abstract").alias("source"),
                pl.lit(0.9).alias("confidence"),
                pl.lit(10).alias("character_offset_start"),
            )
            .collect()
        )
        return flattened_df

    tagger = NerTagger.get_instance(use_llm=True)
    processed_pubs = get_processed_pubs()

    # remove already processed patents
    filtered = patents.filter(~pl.col("publication_number").is_in(processed_pubs))

    # get patent descriptions
    patent_texts = get_patent_descriptions(filtered)

    # extract entities
    entities = tagger.extract(patent_texts, flatten_results=False)

    # add back to orig df
    enriched = filtered.with_columns(pl.Series("entities", entities))

    return format(enriched)


def upsert_annotations(df: pl.DataFrame):
    """
    Inserts annotations into BigQuery table `patents.annotations`

    Args:
        df (pl.DataFrame): DataFrame with columns `publication_number` and `text`
    """
    logging.info(f"Upserting %s", df)

    annotation_df = df.groupby("publication_number").agg(
        pl.struct(
            *[pl.col(name) for name in df.columns if name != "publication_number"]
        ).alias("annotations")
    )

    logging.info(f"Upserting annotations to BigQuery, %s", annotation_df)

    upsert_into_bg_table(
        annotation_df,
        "patents.annotations",
        id_fields=["publication_number"],
        insert_fields=["publication_number", "annotations"],
        on_conflict="target.annotations = ARRAY_CONCAT(target.annotations, source.annotations)",
    )


def upsert_terms(df: pl.DataFrame):
    """
    Upserts `terms` to BigQuery
    """

    terms_df = df.groupby(by=["term", "domain"]).count()

    logging.info(f"Upserting terms to BigQuery, %s", terms_df)
    upsert_into_bg_table(
        terms_df,
        "patents.terms",
        id_fields=["term", "domain"],
        insert_fields=["term", "domain", "count"],
        on_conflict="target.count = target.count + source.count",
    )


def get_processed_pubs() -> list[str]:
    """
    Returns a list of already processed publication numbers
    """
    with open(PROCESSED_PUBS_FILE, "r") as f:
        return f.read().splitlines()


def checkpoint(df: pl.DataFrame, id: Optional[str] = None) -> None:
    """
    Persists processing state
    - processed publication numbers added to a file
    - df written to parquet file (if `id` procided)
    """

    if id:
        filename = f"{BASE_DIR}/chunk_{id}.parquet"
        logging.info(f"Writing df chunk to {filename}")
        df.write_parquet(filename)

    logging.info(f"Persisting processed publication_numbers")
    with open(PROCESSED_PUBS_FILE, "a") as f:
        f.write("\n".join(df["publication_number"].to_list()))


def enrich_with_ner(terms: list[str]) -> None:
    """
    Enriches patents with NER annotations

    - Pulls patents from BigQuery
    - Checks to see if they have already been processed
    - Enrich with NER annotations
    - Persist to annotations and terms tables

    Args:
        terms: list of terms for which to pull and enrich patents
    """
    patents = get_patents(terms, last_id=None)
    last_id = max(patent["publication_number"] for patent in patents)

    while patents:
        df = enrich_patents(pl.DataFrame(patents))

        upsert_annotations(df)
        upsert_terms(df)
        checkpoint(df, get_id([*terms, last_id]))

        patents = get_patents(terms, last_id=last_id)
        if patents:
            last_id = max(patent["publication_number"] for patent in patents)
