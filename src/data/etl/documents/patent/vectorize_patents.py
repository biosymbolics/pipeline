import asyncio
import logging
import sys
from typing import Optional

from constants.core import APPLICATIONS_TABLE, PATENT_VECTOR_TABLE
from data.etl.documents import DocumentVectorizer
from system import initialize


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VECTORIZED_PROCESSED_DOCS_FILE = "data/vectorized_processed_patents.txt"


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
            text_fields=["title", "abstract", "claims"],
            id_field="publication_number",
            processed_docs_file=VECTORIZED_PROCESSED_DOCS_FILE,
            batch_size=1000,
        )

    async def _fetch_batch(
        self, last_id: Optional[str] = None, only_with_claims: bool = False
    ) -> list[dict]:
        """
        Fetch a batch of documents to vectorize

        Args:
            last_id (Optional[str], optional): last id to paginate from. Defaults to None.
            only_with_claims (bool, optional): only fetch documents with claims. Defaults to False.
        """
        claims_where = "AND array_length(claims, 1) > 0" if only_with_claims else ""
        pagination_where = f"AND {self.id_field} > '{last_id}'" if last_id else ""

        query = f"""
            SELECT publication_number, title, abstract, array_to_string(claims, ' ') as claims
            FROM {APPLICATIONS_TABLE}
            WHERE 1 = 1
            {claims_where}
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
            Usage: python3 -m data.etl.documents.patent.vectorize_patents [starting_id] [--only_with_claims]
            """
        )
        sys.exit()

    only_with_claims = "--only_with_claims" in sys.argv
    starting_id = sys.argv[1] if len(sys.argv) > 1 else None
    vectorizer = PatentVectorizer()
    asyncio.run(vectorizer(starting_id, only_with_claims=only_with_claims))
