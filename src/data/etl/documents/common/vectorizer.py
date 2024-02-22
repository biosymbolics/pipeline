from abc import abstractmethod
import logging
import time
from typing import Optional, Sequence, TypedDict
import polars as pl
from pydash import compact, uniq
import hashlib


from clients.low_level.postgres import PsqlDatabaseClient
from core.ner.spacy import get_transformer_nlp


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_TEXT_LENGTH = 2000
VECTORIZED_PROCESSED_DOCS_FILE = "data/vectorized_processed_pubs.txt"


VectorizedDoc = TypedDict("VectorizedDoc", {"id": str, "vector": list[float]})


class DocumentVectorizer:
    """
    Base class for document vectorization
    """

    def __init__(
        self,
        database: str,
        dest_table: str,
        text_fields: Sequence[str],
        id_field: str,
        processed_docs_file: str,
        batch_size: int = 1000,
    ):
        """
        Initialize the vectorizer
        """
        logger.info("Initializing DocumentVectorizer with db %s", database)
        self.db = PsqlDatabaseClient(database)
        self.processed_docs_file = processed_docs_file
        self.batch_size = batch_size
        self.text_fields = text_fields
        self.dest_table = dest_table
        self.id_field = id_field
        self.nlp = get_transformer_nlp()

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
        logger.debug(f"Persisting processed ids (%s)", len(ids))
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
        print(doc_df)
        texts = compact([doc_df[field].to_list() for field in self.text_fields])
        return [("\n".join(text))[0:MAX_TEXT_LENGTH] for text in zip(*texts)]

    def _vectorize(self, documents: list[str]) -> list[list[float]]:
        """
        Vectorize document descriptions

        Uniqs the documents before vectorization to reduce redundant processing

        Note: this appears to have a memory leak. Perhaps it is just Spacy Vocab,
        but it seems like something more.
        """
        uniq_documents = uniq(documents)

        if len(uniq_documents) < len(documents):
            logger.info(
                f"Avoiding processing of %s duplicate documents",
                len(documents) - len(uniq_documents),
            )

        # vectorize
        vectors = [doc.vector.tolist() for doc in self.nlp.pipe(uniq_documents)]

        # map of content hash to vector
        vectors_map = {
            hashlib.sha1(c.encode()).hexdigest(): v
            for c, v in zip(uniq_documents, vectors)
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
        already_processed = self._get_processed_docs()
        to_process = df.filter(~pl.col(self.id_field).is_in(already_processed))

        if len(to_process) == 0:
            logger.info("No documents to process")
            return [], []

        ids = to_process[self.id_field].to_list()

        # generate doc texts
        texts = self._generate_text(to_process)

        return ids, texts

    async def handle_batch(self, batch: list[dict]) -> None:
        start = time.monotonic()

        # preprocess, vectorize
        ids, texts = self._preprocess(batch)
        vectors = self._vectorize(texts)

        # create inserts
        inserts = [VectorizedDoc(id=id, vector=v) for id, v in zip(ids, vectors)]

        # persist doc-level embeddings
        if inserts is not None:
            await self.db.insert_into_table(inserts, self.dest_table)
            self._checkpoint(ids)

        logger.info(
            "Processed %s documents in %s seconds",
            len(ids),
            round(time.monotonic() - start),
        )

    async def __call__(self, starting_id: Optional[str] = None) -> None:
        """
        Vectorize & persist documents
        """

        batch = await self._fetch_batch(last_id=starting_id)
        i = 0

        while batch:
            await self.handle_batch(batch)
            last_id = batch[-1][self.id_field]
            batch = await self._fetch_batch(last_id=str(last_id))

            # clear vocab to reduce memory utilization & force gc
            if (i % 10) == 0:
                logger.info("Clearing vocab (%s)", len(self.nlp.vocab()))
                self.nlp.reset()

            i += 1
