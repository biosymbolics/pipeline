import asyncio
import logging
import sys
from typing import Optional


from constants.core import TRIAL_VECTOR_TABLE
from data.etl.documents import DocumentVectorizer
from system import initialize


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

VECTORIZED_PROCESSED_DOCS_FILE = "data/vectorized_processed_trials.txt"


initialize()


class TrialVectorizer(DocumentVectorizer):
    def __init__(self):
        """
        Initialize the vectorizer

        Before running, ensure the destination table is created:
            create table patent_embeddings (id text, vector vector(768));
        """
        super().__init__(
            database="aact",
            dest_table=TRIAL_VECTOR_TABLE,
            # TODO: add more fields like adverse events, dropouts, inclusion criteria
            text_fields=[
                "description",
                "official_title",
                # "baseline_population",
                "intervention_text",
                "condition_text",
                # "inclusion_criteria",
            ],
            id_field="nct_id",
            processed_docs_file=VECTORIZED_PROCESSED_DOCS_FILE,
            batch_size=1000,
        )

    async def _fetch_batch(self, last_id: Optional[str] = None) -> list[dict]:
        """
        Fetch a batch of documents to vectorize

        Args:
            last_id (Optional[str], optional): last id to paginate from. Defaults to None.
        """
        pagination_where = (
            f"AND studies.{self.id_field} > '{last_id}'" if last_id else ""
        )

        query = f"""
            SELECT
                studies.nct_id as nct_id,
                COALESCE(max(brief_summaries.description), '') as description,
                COALESCE(max(official_title), max(brief_title)) as official_title,
                -- concat('Population: ', max(baseline_population), '.') as baseline_population,
                concat('Interventions: ', string_agg(interventions.name, ','), '.') as intervention_text,
                concat('Indications: ', string_agg(conditions.name, ','), '.') as condition_text,
                concat('Inclusion criteria: ', max(eligibilities.criteria), '.') as inclusion_criteria
            FROM studies, brief_summaries, eligibilities, interventions, conditions
            WHERE studies.nct_id = brief_summaries.nct_id
            AND studies.nct_id = eligibilities.nct_id
            AND studies.nct_id = interventions.nct_id
            AND studies.nct_id = conditions.nct_id
            {pagination_where}
            GROUP BY studies.nct_id
            ORDER BY {self.id_field} ASC
            limit {self.batch_size}
        """
        trials = await self.db.select(query)
        return trials


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.documents.trial.vectorize_trials [starting_id]
            """
        )
        sys.exit()

    starting_id = sys.argv[1] if len(sys.argv) > 1 else None
    vectorizer = TrialVectorizer()
    asyncio.run(vectorizer(starting_id))
