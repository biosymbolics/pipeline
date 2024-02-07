from abc import abstractmethod
import asyncio
import logging
import sys
from typing import Any, Optional, Sequence, TypedDict
import polars as pl
from pydash import uniq
import hashlib

from system import initialize

initialize()


from clients.low_level.postgres import PsqlDatabaseClient
from core.ner.spacy import get_transformer_nlp


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_TEXT_LENGTH = 2000
VECTORIZED_PROCESSED_DOCS_FILE = "data/vectorized_processed_pubs.txt"


VectorizedDoc = TypedDict("VectorizedDoc", {"id": str, "vector": list[float]})


class DocumentVectorizer:
    def __init__(
        self,
        source_table: str,
        dest_table: str,
        text_fields: Sequence[str],
        id_field: str,
        processed_docs_file: str,
        batch_size: int = 1000,
    ):
        """
        Initialize the vectorizer
        """
        self.db = PsqlDatabaseClient("patents")
        self.processed_docs_file = processed_docs_file
        self.batch_size = batch_size
        self.text_fields = text_fields
        self.source_table = source_table
        self.dest_table = dest_table
        self.id_field = id_field

    def _get_processed_docs(self) -> list[str]:
        """
        Returns a list of already processed docs
        """
        try:
            with open(self.processed_docs_file, "r") as f:
                return f.read().splitlines()
        except FileNotFoundError:
            return []

    def _checkpoint(self, ids: Sequence[str]) -> None:
        """
        Persists processing state
        """
        logger.info(f"Persisting processed ids (%s)", len(ids))
        with open(self.processed_docs_file, "a+") as f:
            f.write("\n" + "\n".join(ids))

    @abstractmethod
    async def _fetch_batch(self, last_id: Optional[str] = None) -> list[dict]:
        """
        Fetch a batch of documents to vectorize

        Args:
            last_id (Optional[str], optional): last id to paginate from. Defaults to None.
        """
        raise NotImplementedError

    def _generate_text(self, doc_df: pl.DataFrame) -> list[str]:
        """
        Format documents for vectorization
        Concatenates fields and trucates to MAX_TEXT_LENGTH
        """
        texts = [doc_df[field].to_list() for field in self.text_fields]
        return [("\n".join(text))[0:MAX_TEXT_LENGTH] for text in zip(*texts)]

    def _vectorize(self, documents: list[str]) -> list[list[float]]:
        """
        Vectorize document descriptions

        Uniqs the documents before vectorization to reduce redundant processing
        """
        uniq_documents = uniq(documents)

        if len(uniq_documents) < len(documents):
            logger.info(
                f"Avoiding processing of %s duplicate documents",
                len(documents) - len(uniq_documents),
            )

        # vectorize
        nlp = get_transformer_nlp()
        vectors = [doc.vector.tolist() for doc in list(nlp.pipe(uniq_documents))]

        # map of content hash to vector
        vectors_map = {
            hashlib.sha1(c.encode()).hexdigest(): de
            for c, de in zip(uniq_documents, vectors)
        }

        # return vectors for all provided documents (even dups)
        all_vectors = [
            vectors_map[hashlib.sha1(doc.encode()).hexdigest()] for doc in documents
        ]

        return all_vectors

    def _preprocess(
        self,
        documents: list[dict],
    ) -> tuple[list[str], list[str]]:
        """
        Preprocess documents for vectorization
        """
        df = pl.DataFrame(documents)

        # remove already processed documents
        to_process = df.filter(~pl.col(self.id_field).is_in(self._get_processed_docs()))

        if len(to_process) == 0:
            logger.info("No documents to process")
            return [], []

        ids = to_process[self.id_field].to_list()

        # generate doc texts
        texts = self._generate_text(to_process)

        return ids, texts

    async def __call__(self, starting_id: Optional[str] = None) -> None:
        """
        Vectorize & persist documents
        """
        batch = await self._fetch_batch(last_id=starting_id)

        while batch:
            # get the new last id
            last_id = batch[-1][self.id_field]

            # preprocess, vectorize
            ids, texts = self._preprocess(batch)
            vectors = self._vectorize(texts)

            # create inserts
            inserts = [VectorizedDoc(id=id, vector=v) for id, v in zip(ids, vectors)]

            # persist doc-level embeddings
            if inserts is not None:
                await self.db.insert_into_table(inserts, self.dest_table)
                self._checkpoint(ids)

            batch = await self._fetch_batch(last_id=str(last_id))


class PatentVectorizer(DocumentVectorizer):
    def __init__(self):
        """
        Initialize the vectorizer

        Before running, ensure the destination table is created:
            create table patent_embeddings (id text, vector vector(768));
        """
        super().__init__(
            source_table="applications",
            dest_table="patent_embeddings",
            text_fields=["title", "abstract"],
            id_field="publication_number",
            processed_docs_file=VECTORIZED_PROCESSED_DOCS_FILE,
            batch_size=1000,
        )

    async def _fetch_batch(self, last_id: Optional[str] = None) -> list[dict]:
        """
        Fetch a batch of documents to vectorize

        Args:
            last_id (Optional[str], optional): last id to paginate from. Defaults to None.
        """
        pagination_where = f"AND {self.id_field} > '{last_id}'" if last_id else ""

        query = f"""
            SELECT publication_number, title, abstract
            FROM applications
            WHERE 1 = 1
            {pagination_where}
            ORDER BY {self.id_field} ASC
            limit {self.batch_size}
        """
        patents = await self.db.select(query)
        return patents


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.documents.patent.vectorize_patents [starting_id]
            Loads NER data for patents
            """
        )
        sys.exit()

    starting_id = sys.argv[1] if len(sys.argv) > 1 else None
    vectorizer = PatentVectorizer()
    asyncio.run(vectorizer(starting_id))
