import asyncio
import logging
import sys
from typing import Optional, TypedDict
from constants.core import APPLICATIONS_TABLE, PATENT_VECTOR_TABLE


from data.etl.documents.common.vectorizer import DocumentVectorizer
from system import initialize


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

MAX_TEXT_LENGTH = 2000
VECTORIZED_PROCESSED_DOCS_FILE = "data/vectorized_processed_patents.txt"


VectorizedDoc = TypedDict("VectorizedDoc", {"id": str, "vector": list[float]})

initialize()


class PatentVectorizer(DocumentVectorizer):
    def __init__(self):
        """
        Initialize the vectorizer

        Before running, ensure the destination table is created:
            create table patent_embeddings (id text, vector vector(768));
        """
        super().__init__(
            database="patents",
            dest_table=PATENT_VECTOR_TABLE,
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
            FROM {APPLICATIONS_TABLE}
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
