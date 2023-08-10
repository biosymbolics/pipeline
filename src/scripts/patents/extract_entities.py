import logging
import sys
from typing import Any, Optional
import polars as pl
from pydash import flatten
from data.ner.classifier import classify_by_keywords
from data.ner.types import DocEntities, DocEntity

from system import initialize

initialize()

from constants.patents import PATENT_ATTRIBUTE_MAP
from clients.low_level.big_query import select_from_bg, upsert_df_into_bg_table
from constants.core import SOURCE_BIOSYM_ANNOTATIONS_TABLE
from data.ner import NerTagger


ID_FIELD = "publication_number"
TEXT_FIELDS = ["title", "abstract"]
ENTITY_TYPES = ["compounds", "diseases", "mechanisms"]
BATCH_SIZE = 1000
MAX_TEXT_LENGTH = 500
PROCESSED_PUBS_FILE = "data/processed_pubs.txt"
BASE_DIR = "data/ner_enriched"


class PatentEnricher:
    """
    Enriches patents with NER tags
    """

    def __init__(self):
        """
        Initialize the enricher
        """
        self.tagger = NerTagger.get_instance(entity_types=ENTITY_TYPES)

    def __get_processed_pubs(self) -> list[str]:
        """
        Returns a list of already processed publication numbers
        """
        with open(PROCESSED_PUBS_FILE, "r") as f:
            return f.read().splitlines()

    def __checkpoint(self, df: pl.DataFrame) -> None:
        """
        Persists processing state
        - processed publication numbers added to a file
        """
        logging.info(f"Persisting processed publication_numbers")
        with open(PROCESSED_PUBS_FILE, "a") as f:
            f.write("\n" + "\n".join(df["publication_number"].to_list()))

    def __fetch_patents_batch(self, last_id: Optional[str] = None) -> list[dict]:
        """
        Fetch a batch of patents from BigQuery

        Args:
            last_id (Optional[str], optional): last id to paginate from. Defaults to None.

        TODO: don't depend upon generated table?
        """
        pagination_where = f"AND apps.{ID_FIELD} > '{last_id}'" if last_id else ""

        query = f"""
            SELECT apps.publication_number, apps.title, apps.abstract
            FROM patents.applications AS apps
            WHERE 1 = 1
            {pagination_where}
            ORDER BY apps.{ID_FIELD} ASC
            limit {BATCH_SIZE}
        """
        patents = select_from_bg(query)
        return patents

    def __format_patent_docs(self, patents: pl.DataFrame) -> list[str]:
        """
        Get patent descriptions (title + abstract)
        Concatenates title and abstract into `text` column, trucates to MAX_TEXT_LENGTH
        """
        df = patents.with_columns(
            pl.concat_str(TEXT_FIELDS, separator="\n").alias("text"),
        )

        return [text[0:MAX_TEXT_LENGTH] for text in df["text"].to_list()]

    def __enrich_patents(self, patents: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        Enriches patents with entities

        Args:
            patents (pl.DataFrame): patents to enrich

        Returns:
            pl.DataFrame: enriched patents
        """

        processed_pubs = self.__get_processed_pubs()

        # remove already processed patents
        unprocessed = patents.filter(
            ~pl.col("publication_number").is_in(processed_pubs)
        )

        if len(unprocessed) == 0:
            logging.info("No patents to process")
            return None

        if len(unprocessed) < len(patents):
            logging.info(
                f"Filtered out %s patents that have already been processed",
                len(patents) - len(unprocessed),
            )

        # get patent descriptions
        patent_docs = self.__format_patent_docs(unprocessed)

        # extract entities
        # normalization/linking is unnecessary; will be handled by initialize_patents.
        entities = self.tagger.extract(
            patent_docs,
            link=False,
        )

        patent_attributes: list[DocEntities] = [
            [
                DocEntity(a_set[0], "attribute", 0, 0, a_set[0], None)
                for a_set in attribute_set
            ]
            for attribute_set in classify_by_keywords(patent_docs, PATENT_ATTRIBUTE_MAP)
        ]

        all_entities = [list(set(a + b)) for a, b in zip(entities, patent_attributes)]

        if len(flatten(all_entities)) == 0:
            logging.info("No entities found")
            return None

        # add back to orig df
        enriched = unprocessed.with_columns(pl.Series("entities", all_entities))

        flattened_df = (
            enriched.explode("entities")
            .lazy()
            .select(
                pl.col("publication_number"),
                pl.col("entities").apply(lambda e: e[0]).alias("original_term"),
                pl.col("entities").apply(lambda e: e[1]).alias("domain"),
                pl.lit(0.90000001).alias("confidence"),
                pl.lit("title+abstract").alias("source"),
                pl.col("entities")
                .apply(lambda e: e[3])
                .alias("character_offset_start"),
                pl.col("entities").apply(lambda e: e[4]).alias("character_offset_end"),
            )
            .collect()
        )
        return flattened_df

    def __upsert_biosym_annotations(self, df: pl.DataFrame):
        """
        Upserts to the bs_annotations table (our NER annotations)

        Args:
            df (pl.DataFrame): DataFrame of annotations (from __enrich_patents/__flatten)
        """
        logging.info(f"Upserting %s", df)

        upsert_df_into_bg_table(
            df,
            SOURCE_BIOSYM_ANNOTATIONS_TABLE,
            id_fields=[ID_FIELD, "original_term", "domain"],
            insert_fields=[
                ID_FIELD,
                "original_term",
                "domain",
                "confidence",
                "source",
                "character_offset_start",
                "character_offset_end",
            ],
            on_conflict="UPDATE SET target.domain = source.domain",  # NOOP
        )

    def extract(self, starting_id: Optional[str] = None) -> None:
        """
        Enriches patents with NER annotations

        - Pulls patents from BigQuery
        - Checks to see if they have already been processed
        - Enrich with NER annotations
        - Persist to annotations and terms tables

        Args:
            starting_id (Optional[str], optional): last id to paginate from. Defaults to None.
        """
        patents = self.__fetch_patents_batch(last_id=starting_id)
        last_id = max(patent["publication_number"] for patent in patents)

        while patents:
            df = self.__enrich_patents(pl.DataFrame(patents))

            if df is not None:
                self.__upsert_biosym_annotations(df)
                self.__checkpoint(df)

            patents = self.__fetch_patents_batch(last_id=str(last_id))
            if patents:
                last_id = max(patent["publication_number"] for patent in patents)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.extract(*args, **kwds)


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 ner.py [starting_id]\nLoads NER data for patents")
        sys.exit()

    starting_id = sys.argv[1] if len(sys.argv) > 1 else None
    enricher = PatentEnricher()
    terms = [
        "schizophrenia",
        "pulmonary hypertension",
        "bipolar disorder",
        "depression",
        "major depressive disorder",
        "asthma",
        "melanoma",
        "alzheimer's disease",
    ]
    enricher(None, starting_id)
