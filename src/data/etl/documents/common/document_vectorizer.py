from abc import abstractmethod
from enum import Enum
from functools import reduce
import logging
import time
from typing import Optional, Sequence, TypedDict
import polars as pl
from pydash import compact, flatten, uniq


from clients.low_level.postgres import PsqlDatabaseClient
from nlp.vectorizing import Vectorizer
from data.etl.base_etl import BaseEtl
from typings.core import is_list_string_list, is_string_list
from utils.tensor import l1_regularize, tensor_mean


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_TEXT_LENGTH = 2000
MAX_ARRAY_LENGTH = 10
VECTORIZED_PROCESSED_DOCS_FILE = "data/vectorized_processed_pubs.txt"


VectorizedDoc = TypedDict("VectorizedDoc", {"id": str, "vector": list[float]})


class ComboStrategy(Enum):
    text_concat = "text_concat"
    average = "average"


class DocumentVectorizer(BaseEtl):
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
        combo_strategy: ComboStrategy = ComboStrategy.text_concat,
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
        self.combo_strategy: ComboStrategy = combo_strategy
        self.vectorizer = Vectorizer.get_instance()

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
    async def _fetch_batch(self, last_id: Optional[str] = None, **kwargs) -> list[dict]:
        """
        Fetch a batch of documents to vectorize

        Args:
            last_id (Optional[str], optional): last id to paginate from. Defaults to None.
        """
        raise NotImplementedError

    def _generate_text(self, doc_df: pl.DataFrame) -> list[str] | list[list[str]]:
        """
        Format documents for vectorization
        Concatenates fields and trucates to MAX_TEXT_LENGTH
        """
        texts: list[list[str]] = compact(
            [doc_df[field].to_list() for field in self.text_fields]
        )

        def prep_text_set(text_set: tuple[list[str] | str]) -> list[str]:
            return [
                ", ".join(uniq(ts)[0:MAX_ARRAY_LENGTH]) if isinstance(ts, list) else ts
                for ts in text_set
            ]

        if self.combo_strategy == "text_concat":
            return [
                (". ".join(prep_text_set(text_set)))[0:MAX_TEXT_LENGTH]
                for text_set in zip(*texts)
            ]

        res = [prep_text_set(text_set) for text_set in zip(*texts)]

        if len(doc_df) != len(res):
            raise ValueError("Mismatched text and document lengths")
        return res

    def _vectorize(self, docs: list[str] | list[list[str]]) -> list[list[float]]:
        """
        Vectorize document descriptions
        """
        # if the combo strategy is text_concat, combine list of strings and then vectorize
        if self.combo_strategy == ComboStrategy.text_concat and is_string_list(docs):
            vectors = self.vectorizer.vectorize(docs)
            return [l1_regularize(v).tolist() for v in vectors]

        # if the combo strategy is average, vectorize each string and average
        elif self.combo_strategy == ComboStrategy.average and is_list_string_list(docs):
            indices: list[tuple[int, int]] = reduce(
                lambda acc, d: acc + [(acc[-1][1], acc[-1][1] + len(d))],
                docs,
                [(0, 0)],  # type: ignore
            )
            vectors = self.vectorizer.vectorize(flatten(docs))
            vector_sets = [
                l1_regularize(tensor_mean(vectors[s:e])).tolist()
                for s, e in indices[1:]
            ]

            if indices[-1][1] != len(flatten(docs)):
                raise ValueError("Mismatched vector and document lengths")

            return vector_sets

        raise ValueError("Invalid combination strategy or datatype")

    def _preprocess(
        self,
        documents: list[dict],
    ) -> tuple[list[str], list[str] | list[list[str]]]:
        """
        Preprocess documents for vectorization
        """
        df = pl.DataFrame(documents)

        # remove already processed documents
        already_processed = self._get_processed_docs()
        to_process = df.filter(~pl.col(self.id_field).is_in(already_processed))

        if len(to_process) == 0:
            logger.info("No documents to process")
            return [], []  # type: ignore

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

    async def __call__(
        self, starting_id: Optional[str] = None, **fetch_batch_args
    ) -> None:
        """
        Vectorize & persist documents
        """

        if starting_id is None and len(self._get_processed_docs()) == 0:
            logger.warning(
                "No starting_id and no processed docs; clearing vector table if exists"
            )
            await self.db.execute_query(f"TRUNCATE {self.dest_table}")

        batch = await self._fetch_batch(last_id=starting_id, **fetch_batch_args)
        i = 0

        while batch:
            await self.handle_batch(batch)
            last_id = batch[-1][self.id_field]
            batch = await self._fetch_batch(last_id=str(last_id))

            if i % 10 == 0:
                logger.info("Processed %s batches; emptying cache", i)
                self.vectorizer.empty_cache()

            i += 1
