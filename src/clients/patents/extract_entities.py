import logging
from typing import Any, Optional
import polars as pl
from clients.low_level.big_query import select_from_bg, upsert_into_bg_table

from common.ner.ner import NerTagger

ID_FIELD = "publication_number"
CHUNK_SIZE = 500

MAX_TEXT_LENGTH = 500
DECAY_RATE = 1 / 2000
PROCESSED_PUBS_FILE = "data/processed_pubs.txt"
BASE_DIR = "data/ner_enriched"
MIN_SEARCH_RANK = 0.1


class PatentEnricher:
    """
    Enriches patents with NER tags
    """

    def __init__(self):
        """
        Initialize the enricher
        """
        self.tagger = NerTagger.get_instance(use_llm=False, content_type="text")

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
            f.write("\n".join(df["publication_number"].to_list()))

    def __fetch_patents_batch(
        self, terms: list[str], last_id: Optional[str] = None
    ) -> list[dict]:
        """
        Fetch a batch of patents from BigQuery

        Args:
            terms (list[str]): terms on which to search for patents
            last_id (Optional[str], optional): last id to paginate from. Defaults to None.

        TODO: don't depend upon generated table?
        """
        lower_terms = [term.lower() for term in terms]

        pagination_where = f"AND apps.{ID_FIELD} > '{last_id}'" if last_id else ""

        query = f"""
            WITH matches AS (
                SELECT
                    a.publication_number as publication_number,
                    AVG(EXP(-annotation.character_offset_start * {DECAY_RATE})) as search_rank, --- exp decay scaling; higher is better
                FROM patents.annotations a,
                UNNEST(a.annotations) as annotation
                WHERE annotation.term IN UNNEST({lower_terms})
                GROUP BY publication_number
            )
            SELECT apps.publication_number, apps.title, apps.abstract
            FROM patents.applications AS apps, matches
            WHERE apps.publication_number = matches.publication_number
            AND search_rank > {MIN_SEARCH_RANK}
            {pagination_where}
            ORDER BY apps.{ID_FIELD} ASC
            limit {CHUNK_SIZE}
        """
        patents = select_from_bg(query)
        return patents

    def __format_patent_docs(self, patents: pl.DataFrame) -> list[str]:
        """
        Get patent descriptions (title + abstract)
        Concatenates title and abstract into `text` column, trucates to MAX_TEXT_LENGTH
        """
        df = patents.with_columns(
            pl.concat_str(["title", "abstract"], separator="\n").alias("text"),
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

        def __flatten(df: pl.DataFrame):
            """
            Unpacks entities into separate rows
            """
            flattened_df = (
                df.explode("entities")
                .lazy()
                .select(
                    pl.col("publication_number"),
                    pl.col("entities").apply(lambda e: e[3]).alias("canonical_term"),
                    pl.col("entities").apply(lambda e: e[2]).alias("canonical_id"),
                    pl.col("entities").apply(lambda e: e[0]).alias("original_term"),
                    pl.col("entities").apply(lambda e: e[1]).alias("domain"),
                    pl.lit(0.90000001).alias("confidence"),
                    pl.lit("title+abstract").alias("source"),
                    pl.lit(10).alias("character_offset_start"),
                )
                .collect()
                .filter(pl.col("canonical_id").is_not_null())
            )
            return flattened_df

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
        entities = self.tagger.extract(patent_docs, link=False)
        if len([ent for ent in entities if len(ent) > 0]) == 0:
            logging.info("No entities found")
            return None

        # add back to orig df
        flatish_ents = [
            [
                (ent[0], ent[1], ent[2].id, ent[2].name)
                for ent in ent_set
                if ent[2] is not None
            ]
            for ent_set in entities
        ]
        enriched = unprocessed.with_columns(pl.Series("entities", flatish_ents))

        return __flatten(enriched)

    def __upsert_biosym_annotations(self, df: pl.DataFrame):
        """
        Upserts to the bs_annotations table (our NER annotations)

        Args:
            df (pl.DataFrame): DataFrame of annotations (from __enrich_patents/__flatten)
        """
        logging.info(f"Upserting %s", df)

        upsert_into_bg_table(
            df,
            "biosym_annotations",
            id_fields=[ID_FIELD, "original_term", "domain", "canonical_id"],
            insert_fields=[
                ID_FIELD,
                "canonical_term",
                "canonical_id",
                "original_term",
                "domain",
                "confidence",
                "source",
                "character_offset_start",
            ],
            on_conflict="UPDATE SET target.domain = source.domain",  # NOOP
        )

    def extract(self, terms: list[str]) -> None:
        """
        Enriches patents with NER annotations

        - Pulls patents from BigQuery
        - Checks to see if they have already been processed
        - Enrich with NER annotations
        - Persist to annotations and terms tables

        Args:
            terms: list of terms for which to pull and enrich patents
        """
        patents = self.__fetch_patents_batch(terms, last_id=None)
        last_id = max(patent["publication_number"] for patent in patents)

        while patents:
            df = self.__enrich_patents(pl.DataFrame(patents))

            if df is not None:
                self.__upsert_biosym_annotations(df)
                self.__checkpoint(df)

            patents = self.__fetch_patents_batch(terms, last_id=last_id)
            if patents:
                last_id = max(patent["publication_number"] for patent in patents)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.extract(*args, **kwds)
