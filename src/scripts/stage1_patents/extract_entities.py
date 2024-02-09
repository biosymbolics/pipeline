from abc import abstractmethod
import asyncio
import logging
import sys
from typing import Any, Optional, TypedDict
import polars as pl
from pydash import flatten, uniq
import hashlib

from system import initialize

initialize()

from clients.low_level.database import DatabaseClient
from clients.low_level.big_query import BQDatabaseClient
from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import SOURCE_BIOSYM_ANNOTATIONS_TABLE
from constants.patents import ATTRIBUTE_FIELD, get_patent_attribute_map
from constants.umls import NER_ENTITY_TYPES
from core.ner.classifier import classify_by_keywords
from core.ner.types import DocEntities, DocEntity
from core.ner import NerTagger
from utils.classes import overrides


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ID_FIELD = "publication_number"
MAX_TEXT_LENGTH = 2000
ENRICH_PROCESSED_PUBS_FILE = "data/enrich_processed_pubs.txt"
CLASSIFY_PROCESSED_PUBS_FILE = "data/classify_processed_pubs.txt"


EmbeddingData = TypedDict(
    "EmbeddingData", {"publication_number": str, "vector": list[float]}
)


def extract_attributes(patent_docs: list[str]) -> list[DocEntities]:
    attr_map = get_patent_attribute_map()
    return [
        [DocEntity.create(term, 0, 0, term, ATTRIBUTE_FIELD) for term in attribute_set]
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
        ids = uniq(df["publication_number"].to_list())
        logger.info(f"Persisting processed publication_numbers (%s)", len(ids))
        with open(self.processed_pubs_file, "a+") as f:
            f.write("\n" + "\n".join(ids))

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

    def _extract(
        self, patents: list[dict]
    ) -> tuple[pl.DataFrame, list[EmbeddingData]] | None:
        """
        Enriches patents with entities

        Args:
            patents (pl.DataFrame): patents to enrich

        Returns:
            tuple[pl.DataFrame, list[dict]] | None: entities, embeddings
        """

        df = pl.DataFrame(patents)

        # remove already processed patents
        patents_to_process = df.filter(
            ~pl.col("publication_number").is_in(self._get_processed_pubs())
        )
        patent_ids = patents_to_process["publication_number"].to_list()

        if len(patents_to_process) == 0:
            logger.info("No patents to process")
            return None

        if len(patents_to_process) < len(patents):
            logger.warning(
                f"Filtered out %s patents that have already been processed",
                len(patents) - len(patents_to_process),
            )

        # get patent descriptions
        patent_docs = self._format_patent_docs(patents_to_process)

        # extract entities
        entities = self.extractor(patent_docs)

        if len(flatten(entities)) == 0:
            logger.warning("No entities found")
            return None

        doc_vector_inserts: list[EmbeddingData] = [
            EmbeddingData(publication_number=id, vector=de)
            for id, de in zip(
                patent_ids,
                [es[0].doc_vector if len(es) > 0 else None for es in entities],
            )
            if de is not None
        ]

        # turn into dicts for polars' sake
        entity_dicts = [
            [
                {
                    **a.to_flat_dict(),
                    "publication_number": patent_ids[i],
                }
                for a in es
            ]
            for i, es in enumerate(entities)
        ]

        entity_df = (
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

        return entity_df, doc_vector_inserts

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

        while patents:
            # get the new last id
            last_id = patents[-1]["publication_number"]

            # do NER
            ent_df, doc_embeddings = self._extract(patents) or (
                None,
                None,
            )

            # persist doc-level embeddings
            if doc_embeddings is not None:
                await self.db.insert_into_table(doc_embeddings, "patent_embeddings")

            # persist entity-level embeddings
            if ent_df is not None:
                await self.upsert(ent_df)
                self._checkpoint(ent_df)

            patents = await self._fetch_patents_batch(last_id=str(last_id))

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

        batch_size = 1000
        super().__init__(ENRICH_PROCESSED_PUBS_FILE, BQDatabaseClient, batch_size)
        self.db = BQDatabaseClient()
        self.tagger = NerTagger.get_instance(
            entity_types=NER_ENTITY_TYPES, link=False, normalize=False, rule_sets=[]
        )
        raise Exception(
            "Switch to postgres before running again, lest we pay $400 again for export"
        )

    @overrides(BaseEnricher)
    async def upsert(self, df: pl.DataFrame):
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
        uniq_content = uniq(patent_docs)

        if len(uniq_content) < len(patent_docs):
            logger.info(
                f"Avoiding processing of %s duplicate patents",
                len(patent_docs) - len(uniq_content),
            )

        entities = self.tagger.extract(uniq_content)
        hash_entities_map = {
            hashlib.sha1(c.encode()).hexdigest(): de
            for c, de in zip(uniq_content, entities)
        }

        all_entities = [
            hash_entities_map[hashlib.sha1(doc.encode()).hexdigest()]
            for doc in patent_docs
        ]

        # attributes = extract_attributes(patent_docs)
        # all = [e[0] + e[1] for e in zip(entities, attributes)]
        return all_entities


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.stage1_patents.extract_entities [starting_id]
            Loads NER data for patents
            """
        )
        sys.exit()

    if True:
        raise Exception(
            "Switch to postgres before running again, lest we pay $400 again for export"
        )

    starting_id = sys.argv[1] if len(sys.argv) > 1 else None
    enricher = PatentEnricher()
    # enricher = PatentClassifier() # only use if wanting to re-classify (comparatively fast)
    asyncio.run(enricher(starting_id))
