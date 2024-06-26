"""
Script for running inference with the trial characteristics model,
which persists these attributes to a table for later use.
"""

import asyncio
import sys
import logging

from clients.low_level.postgres import PsqlDatabaseClient
from system import initialize

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

initialize()


def predict_clindev():
    """
    Using torch clindev model, run inference to predict clindev characteristics
    """
    pass


async def create_predicted_clindev_table():
    """
    Create patent to predicted clindev table
    """
    client = PsqlDatabaseClient()
    table = "predicted_clindev"
    query = """
        select patents.publication_number, nct_id
        from
        aggregated_annotations annotations,
        applications patents
        where patents.publication_number=annotations.publication_number
    """
    await client.create_from_select(query, table)
    await client.create_indices(
        [
            {
                "table": table,
                "column": "publication_number",
            },
        ]
    )


def predict_clindev_characteristics():
    """
    Predict clindev characteristics
    """
    asyncio.run(create_predicted_clindev_table())


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
              Usage: python3 -m scripts.ctgov.copy_ctgov
              Copies ctgov to patents
        """
        )
        sys.exit()

    predict_clindev_characteristics()
