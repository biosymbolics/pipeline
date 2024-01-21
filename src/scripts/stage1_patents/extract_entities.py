from abc import abstractmethod
import asyncio
import logging
import sys
from typing import Any, Optional
import polars as pl
from pydash import flatten


from system import initialize

initialize()

from clients.low_level.database import DatabaseClient
from clients.low_level.big_query import BQDatabaseClient
from clients.low_level.postgres import PsqlDatabaseClient
from constants.patents import ATTRIBUTE_FIELD, get_patent_attribute_map
from constants.core import SOURCE_BIOSYM_ANNOTATIONS_TABLE
from core.ner.classifier import classify_by_keywords
from core.ner.types import DocEntities, DocEntity
from core.ner import NerTagger
from utils.classes import overrides


ID_FIELD = "publication_number"
MAX_TEXT_LENGTH = 2000
ENRICH_PROCESSED_PUBS_FILE = "data/enrich_processed_pubs.txt"
CLASSIFY_PROCESSED_PUBS_FILE = "data/classify_processed_pubs.txt"


ENTITY_TYPES = frozenset(
    [
        "biologics",
        "compounds",
        "devices",
        "diagnotics",
        "diseases",
        "mechanisms",
        "procedures",
    ]
)


def extract_attributes(patent_docs: list[str]) -> list[DocEntities]:
    attr_map = get_patent_attribute_map()
    return [
        [DocEntity(term, 0, 0, term, ATTRIBUTE_FIELD) for term in attribute_set]
        for attribute_set in classify_by_keywords(patent_docs, attr_map)
    ]


class BaseEnricher:
    def __init__(
        self,
        processed_pubs_file: str,
        DbImpl: type[DatabaseClient],
        batch_size: int,
    ):
        """
        Initialize the enricher
        """
        self.db = DbImpl()
        self.processed_pubs_file = processed_pubs_file
        self.batch_size = batch_size

    def _get_processed_pubs(self) -> list[str]:
        """
        Returns a list of already processed publication numbers
        """
        try:
            with open(self.processed_pubs_file, "r") as f:
                return f.read().splitlines()
        except FileNotFoundError:
            return []

    def _checkpoint(self, df: pl.DataFrame) -> None:
        """
        Persists processing state
        - processed publication numbers added to a file
        """
        logging.info(f"Persisting processed publication_numbers")
        with open(self.processed_pubs_file, "a+") as f:
            f.write("\n" + "\n".join(df["publication_number"].to_list()))

    async def _fetch_patents_batch(self, last_id: Optional[str] = None) -> list[dict]:
        """
        Fetch a batch of patents from BigQuery

        Args:
            last_id (Optional[str], optional): last id to paginate from. Defaults to None.
        """
        pagination_where = f"AND apps.{ID_FIELD} > '{last_id}'" if last_id else ""

        table = self.db.get_table_id("applications")

        query = f"""
            SELECT apps.publication_number, apps.title, apps.abstract
            FROM {table} AS apps
            WHERE 1 = 1
            {pagination_where}
            ORDER BY apps.{ID_FIELD} ASC
            limit {self.batch_size}
        """
        patents = await self.db.select(query)
        return patents

    def _format_patent_docs(self, patents: pl.DataFrame) -> list[str]:
        """
        Get patent descriptions (title + abstract)
        Concatenates title and abstract into `text` column, trucates to MAX_TEXT_LENGTH
        """
        titles = patents["title"].to_list()
        abstracts = patents["abstract"].to_list()

        def format(title, abstract) -> str:
            text = "\n".join([title, abstract])
            return text[0:MAX_TEXT_LENGTH]

        texts = [format(title, abstract) for title, abstract in zip(titles, abstracts)]

        return texts

    def _extract(self, patents: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        Enriches patents with entities

        Args:
            patents (pl.DataFrame): patents to enrich

        Returns:
            pl.DataFrame: enriched patents
        """

        processed_pubs = self._get_processed_pubs()

        # remove already processed patents
        unprocessed_patents = patents.filter(
            ~pl.col("publication_number").is_in(processed_pubs)
        )

        if len(unprocessed_patents) == 0:
            logging.info("No patents to process")
            return None

        if len(unprocessed_patents) < len(patents):
            logging.info(
                f"Filtered out %s patents that have already been processed",
                len(patents) - len(unprocessed_patents),
            )

        # get patent descriptions
        patent_docs = self._format_patent_docs(unprocessed_patents)

        # extract entities
        entities = self.extractor(patent_docs)

        if len(flatten(entities)) == 0:
            logging.info("No entities found")
            return None

        unprocessed_patent_ids = unprocessed_patents["publication_number"].to_list()

        # turn into dicts for polars' sake
        entity_dicts = [
            [
                {
                    **a.to_flat_dict(),
                    "publication_number": unprocessed_patent_ids[i],
                }
                for a in es
            ]
            for i, es in enumerate(entities)
        ]

        # TODO: probably some to suppress?
        flattened_df = (
            pl.DataFrame(flatten(entity_dicts))
            .rename(
                {
                    "type": "domain",
                    "start_char": "character_offset_start",
                    "end_char": "character_offset_end",
                }
            )
            .drop(["normalized_term", "canonical_entity"])
            .with_columns(
                pl.lit("title+abstract").alias("source"),
            )
        )

        return flattened_df

    @abstractmethod
    async def upsert(self, df: pl.DataFrame):
        """override with impl"""

    @abstractmethod
    def extractor(self, patent_docs: list[str]) -> list[DocEntities]:
        """override with impl"""

    async def enrich(self, starting_id: Optional[str] = None) -> None:
        """
        Enriches patents with NER annotations

        Args:
            starting_id (Optional[str], optional): last id to paginate from. Defaults to None.
        """
        patents = await self._fetch_patents_batch(last_id=starting_id)
        last_id = max(patent["publication_number"] for patent in patents)

        while patents:
            df = self._extract(pl.DataFrame(patents))

            if df is not None:
                await self.upsert(df)
                self._checkpoint(df)

            patents = await self._fetch_patents_batch(last_id=str(last_id))
            if patents:
                last_id = max(patent["publication_number"] for patent in patents)

    async def __call__(self, *args: Any, **kwds: Any) -> Any:
        await self.enrich(*args, **kwds)


class PatentClassifier(BaseEnricher):
    """
    Enriches patents with classified attributes
    """

    def __init__(self):
        """
        Initialize the enricher
        """
        batch_size = 20000
        super().__init__(CLASSIFY_PROCESSED_PUBS_FILE, PsqlDatabaseClient, batch_size)

    @overrides(BaseEnricher)
    async def upsert(self, df: pl.DataFrame):
        """
        Persist attribute annotations to table

        *** assumes domain="attributes" has been removed first ***
        """
        await self.db.insert_into_table(df.to_dicts(), "annotations")  # ??

    @overrides(BaseEnricher)
    def extractor(self, patent_docs: list[str]) -> list[DocEntities]:
        return extract_attributes(patent_docs)


class PatentEnricher(BaseEnricher):
    """
    Enriches patents with NER tags

    ** runs on bigquery database (but can be moved) **
    """

    def __init__(self):
        """
        Initialize the enricher
        """
        batch_size = 50
        super().__init__(ENRICH_PROCESSED_PUBS_FILE, BQDatabaseClient, batch_size)
        self.db = BQDatabaseClient()
        self.tagger = NerTagger.get_instance(
            entity_types=ENTITY_TYPES, link=False, normalize=False
        )

    @overrides(BaseEnricher)
    async def upsert(self, df: pl.DataFrame):
        print(df)
        await self.db.upsert_df_into_table(
            df,
            SOURCE_BIOSYM_ANNOTATIONS_TABLE,
            id_fields=[
                ID_FIELD,
                "term",
                "domain",
                "source",
                "character_offset_start",
                "character_offset_end",
            ],
            insert_fields=[
                ID_FIELD,
                "term",
                "domain",
                "vector",
                "source",
                "character_offset_start",
                "character_offset_end",
            ],
            on_conflict="UPDATE SET target.domain = source.domain",
        )

    def extractor(self, patent_docs: list[str]) -> list[DocEntities]:
        entities = self.tagger.extract(patent_docs)
        # attributes = extract_attributes(patent_docs)
        # all = [e[0] + e[1] for e in zip(entities, attributes)]
        return entities


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.stage1_patents.extract_entities [starting_id]
            Loads NER data for patents
            """
        )
        sys.exit()

    starting_id = sys.argv[1] if len(sys.argv) > 1 else None
    enricher = PatentEnricher()
    # enricher = PatentClassifier() # only use if wanting to re-classify (comparatively fast)
    asyncio.run(enricher(starting_id))
