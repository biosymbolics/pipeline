import sys
from typing import Sequence
import polars as pl
import logging

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from data.prediction.clindev.predictor import predict
from utils.list import batch, dedup


MAX_TERMS_PER_DOMAIN = 10
TRIAL_PREDICTIONS_TABLE = "patent_clindev_predictions"
BATCH_SIZE = 20000

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def filter_terms_by_domain(rec, domains: Sequence[str]) -> list[str]:
    """
    Filter terms by domain
    (also dedups)
    """
    terms = [z[0] for z in list(zip(rec["terms"], rec["domains"])) if z[1] in domains][
        0:MAX_TERMS_PER_DOMAIN
    ]
    return dedup(terms)


def generate_trial_predictions():
    """
    Generate trial predictions and save 'em.
    """
    client = PsqlDatabaseClient()

    patents = client.select(
        """
        SELECT
            apps.publication_number as publication_number,
            (apps.assignees)[1] as sponsor,
            array_agg(term) as terms,
            array_agg(domain) as domains,
            (array_agg(t.phase ORDER BY t.start_date desc))[1] as starting_phase
        FROM annotations a, applications apps
        LEFT JOIN patent_to_trial ptt on ptt.publication_number = apps.publication_number
        LEFT JOIN trials t on t.nct_id = ptt.nct_id
        where a.publication_number = apps.publication_number
        and domain in ('diseases', 'mechanisms', 'biologics', 'compounds')
        group by apps.publication_number, apps.assignees
        order by apps.publication_number desc
    """
    )

    df = pl.from_dicts(patents, infer_schema_length=None)

    df = (
        df.with_columns(
            *[
                df.select(
                    pl.struct(["terms", "domains"])
                    .apply(lambda rec: filter_terms_by_domain(rec, domains))
                    .alias(t)
                ).to_series()
                for t, domains in {
                    "conditions": ["diseases"],
                    "interventions": ["mechanisms", "biologics", "compounds"],
                }.items()
            ]
        )
        .drop("terms", "domains")
        .filter(
            (pl.col("conditions").arr.lengths() > 0)
            & (pl.col("interventions").arr.lengths() > 0)
        )
    )

    batches = batch(df.to_dicts(), BATCH_SIZE)

    for i, b in enumerate(batches):
        logger.info("Starting batch %s", i)
        predictions = predict(b)

        if i == 0:
            client.create_and_insert(
                TRIAL_PREDICTIONS_TABLE, predictions, batch_size=BATCH_SIZE
            )
        else:
            client._insert(TRIAL_PREDICTIONS_TABLE, predictions)


def main():
    generate_trial_predictions()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.trial_predictions.load_predictions
            """
        )
        sys.exit()

    main()
