from abc import abstractmethod
import logging
import sys
from typing import Any, Callable, Optional
import polars as pl
from pydash import flatten


from system import initialize

initialize()


from clients.low_level.big_query import BQDatabaseClient
from clients.low_level.postgres import PsqlDatabaseClient
from constants.patents import ATTRIBUTE_FIELD, get_patent_attribute_map
from constants.core import SOURCE_BIOSYM_ANNOTATIONS_TABLE
from core.ner.classifier import classify_by_keywords
from core.ner.types import DocEntities, DocEntity
from core.ner import NerTagger
from utils.classes import overrides


ID_FIELD = "publication_number"
ENTITY_TYPES = frozenset(["compounds", "diseases", "mechanisms"])
MAX_TEXT_LENGTH = 2000
ENRICH_PROCESSED_PUBS_FILE = "data/enrich_processed_pubs.txt"
CLASSIFY_PROCESSED_PUBS_FILE = "data/classify_processed_pubs.txt"
BASE_DIR = "data/ner_enriched"


def extract_attributes(patent_docs: list[str]) -> list[DocEntities]:
    attr_map = get_patent_attribute_map()
    return [
        [
            DocEntity(a_set, ATTRIBUTE_FIELD, 0, 0, a_set, None)
            for a_set in attribute_set
        ]
        for attribute_set in classify_by_keywords(patent_docs, attr_map)
    ]


class BaseEnricher:
    def __init__(
        self,
        processed_pubs_file: str,
        DbImpl: Callable[[], Any],
        batch_size: int,
    ):
        """
        Initialize the enricher
        """
        self.db = DbImpl()
        self.processed_pubs_file = processed_pubs_file
        self.batch_size = batch_size

    def __get_processed_pubs(self) -> list[str]:
        """
        Returns a list of already processed publication numbers
        """
        try:
            with open(self.processed_pubs_file, "r") as f:
                return f.read().splitlines()
        except FileNotFoundError:
            return []

    def __checkpoint(self, df: pl.DataFrame) -> None:
        """
        Persists processing state
        - processed publication numbers added to a file
        """
        logging.info(f"Persisting processed publication_numbers")
        with open(self.processed_pubs_file, "a+") as f:
            f.write("\n" + "\n".join(df["publication_number"].to_list()))

    def __fetch_patents_batch(self, last_id: Optional[str] = None) -> list[dict]:
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
        patents = self.db.select(query)
        return patents

    def __format_patent_docs(self, patents: pl.DataFrame) -> list[str]:
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

    def __extract(self, patents: pl.DataFrame) -> Optional[pl.DataFrame]:
        """
        Enriches patents with entities

        Args:
            patents (pl.DataFrame): patents to enrich

        Returns:
            pl.DataFrame: enriched patents
        """

        processed_pubs = self.__get_processed_pubs()

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
        patent_docs = self.__format_patent_docs(unprocessed_patents)

        # extract entities
        entities = self.extractor(patent_docs)

        if len(flatten(entities)) == 0:
            logging.info("No entities found")
            return None

        # turn into dicts for polars' sake
        entity_dicts = pl.Series(
            "entities", ([[a.to_flat_dict() for a in es] for es in entities])
        )

        flattened_df = (
            # polars can't handle explode list[struct] as of 08/14/23
            unprocessed_patents.with_columns(entity_dicts)
            .filter(pl.col("entities").list.lengths() > 0)
            .explode("entities")
            .lazy()
            .select(
                pl.col("publication_number"),
                pl.col("entities").map_elements(lambda e: e["term"]).alias("original_term"),  # type: ignore
                pl.col("entities").map_elements(lambda e: e["normalized_term"]).alias("term"),  # type: ignore
                pl.col("entities").map_elements(lambda e: e["type"]).alias("domain"),  # type: ignore
                pl.lit("title+abstract").alias("source"),
                pl.col("entities")
                .map_elements(lambda e: e["start_char"])  # type: ignore
                .alias("character_offset_start"),
                pl.col("entities")
                .map_elements(lambda e: e["end_char"])  # type: ignore
                .alias("character_offset_end"),
            )
            .collect()
        )
        return flattened_df

    @abstractmethod
    def upsert(self, df: pl.DataFrame):
        """override with impl"""

    @abstractmethod
    def extractor(self, patent_docs: list[str]) -> list[DocEntities]:
        """override with impl"""

    def enrich(self, starting_id: Optional[str] = None) -> None:
        """
        Enriches patents with NER annotations

        Args:
            starting_id (Optional[str], optional): last id to paginate from. Defaults to None.
        """
        patents = self.__fetch_patents_batch(last_id=starting_id)
        last_id = max(patent["publication_number"] for patent in patents)

        while patents:
            df = self.__extract(pl.DataFrame(patents))

            if df is not None:
                self.upsert(df)
                self.__checkpoint(df)

            patents = self.__fetch_patents_batch(last_id=str(last_id))
            if patents:
                last_id = max(patent["publication_number"] for patent in patents)

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        self.enrich(*args, **kwds)


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
    def upsert(self, df: pl.DataFrame):
        """
        Persist attribute annotations to table

        *** assumes domain="attributes" has been removed first ***
        """
        self.db.insert_into_table(df.to_dicts(), "annotations")  # ??

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
        batch_size = 1000
        super().__init__(ENRICH_PROCESSED_PUBS_FILE, BQDatabaseClient, batch_size)
        self.tagger = NerTagger.get_instance(entity_types=ENTITY_TYPES, link=False)

    @overrides(BaseEnricher)
    def upsert(self, df: pl.DataFrame):
        self.db.upsert_df_into_table(
            df,
            SOURCE_BIOSYM_ANNOTATIONS_TABLE,
            id_fields=[
                ID_FIELD,
                "term",
                "original_term",
                "domain",
                "source",
                "character_offset_start",
                "character_offset_end",
            ],
            insert_fields=[
                ID_FIELD,
                "term",
                "original_term",
                "domain",
                # "confidence",
                "source",
                "character_offset_start",
                "character_offset_end",
            ],
            on_conflict="UPDATE SET target.domain = source.domain",
        )

    def extractor(self, patent_docs: list[str]) -> list[DocEntities]:
        entities = self.tagger.extract(patent_docs)
        attributes = extract_attributes(patent_docs)
        all = [e[0] + e[1] for e in zip(entities, attributes)]
        return all


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.patents.extract_entities [starting_id]\nLoads NER data for patents
            """
        )
        sys.exit()

    starting_id = sys.argv[1] if len(sys.argv) > 1 else None
    enricher = PatentEnricher()
    # enricher = PatentClassifier() # only use if wanting to re-classify (comparatively fast)
    enricher(starting_id)
