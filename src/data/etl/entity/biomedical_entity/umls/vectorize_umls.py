import asyncio
import logging
import sys
from typing import Optional

from constants.core import UMLS_VECTOR_TABLE
from data.etl.documents import DocumentVectorizer
from data.etl.entity.biomedical_entity.umls.load_umls import UmlsLoader
from system import initialize


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VECTORIZED_PROCESSED_DOCS_FILE = "data/vectorized_processed_umls.txt"


initialize()


class UmlsVectorizer(DocumentVectorizer):
    def __init__(self):
        """
        Initialize the vectorizer

        Before running, ensure the destination table is created:
            create table patent_embeddings (id text, vector vector(768));
        """
        super().__init__(
            database="umls",
            dest_table=UMLS_VECTOR_TABLE,
            text_fields=["name"],
            id_field="id",
            processed_docs_file=VECTORIZED_PROCESSED_DOCS_FILE,
            batch_size=10000,
        )

    async def _fetch_batch(self, last_id: Optional[str] = None) -> list[dict]:
        """
        Fetch a batch of Umls records to vectorize

        Args:
            last_id (Optional[str], optional): last id to paginate from. Defaults to None.
        """
        pagination_where = [f"{self.id_field} > '{last_id}'"] if last_id else None
        source_sql = UmlsLoader.get_source_sql(pagination_where, self.batch_size)
        umls = await self.db.select(source_sql)
        return umls


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.entity.biomedical_entity.umls.vectorize_umls [starting_id]
            """
        )
        sys.exit()

    starting_id = sys.argv[1] if len(sys.argv) > 1 else None
    vectorizer = UmlsVectorizer()
    asyncio.run(vectorizer(starting_id))
