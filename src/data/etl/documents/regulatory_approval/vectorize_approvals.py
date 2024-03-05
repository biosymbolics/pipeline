import asyncio
import logging
import sys
from typing import Optional


from constants.core import REGULATORY_APPROVAL_VECTOR_TABLE
from data.etl.documents import DocumentVectorizer
from system import initialize

from .load_regulatory_approval import RegulatoryApprovalLoader


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VECTORIZED_PROCESSED_DOCS_FILE = "data/vectorized_processed_approvals.txt"

initialize()


class ApprovalVectorizer(DocumentVectorizer):
    def __init__(self):
        """
        Initialize the vectorizer
        """
        super().__init__(
            database="drugcentral",
            dest_table=REGULATORY_APPROVAL_VECTOR_TABLE,
            text_fields=[
                "label_text",
            ],
            id_field="id",
            processed_docs_file=VECTORIZED_PROCESSED_DOCS_FILE,
            batch_size=3000,
        )

    async def _fetch_batch(self, last_id: Optional[str] = None) -> list[dict]:
        """
        Fetch a batch of documents to vectorize

        Args:
            last_id (Optional[str], optional): last id to paginate from. Defaults to None.
        """
        pagination_where = (
            f"WHERE approvals.{self.id_field} > '{last_id}'" if last_id else ""
        )

        query = RegulatoryApprovalLoader.get_source_sql(
            ["prod.ndc_product_code as id", "MAX(label_text) as label_text"]
        )

        query = f"""
            SELECT *
            FROM ({query}) as approvals
            {pagination_where}
            ORDER BY {self.id_field} ASC
            limit {self.batch_size}
        """
        approvals = await self.db.select(query)
        return approvals


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.documents.regulatory_approval.vectorize_approvals [starting_id]
            """
        )
        sys.exit()

    starting_id = sys.argv[1] if len(sys.argv) > 1 else None
    vectorizer = ApprovalVectorizer()
    asyncio.run(vectorizer(starting_id))
