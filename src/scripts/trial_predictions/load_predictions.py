import sys
from typing import Sequence
import polars as pl

from clients.low_level.postgres.postgres import PsqlDatabaseClient
from data.prediction.clindev.predictor import predict
from utils.list import dedup


def transform_predictions(rows) -> list[dict]:
    """
    Transform
    """

    return rows


MAX_TERMS_PER_DOMAIN = 10


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
            array_agg(term) as terms,
            array_agg(domain) as domains,
            (array_agg(t.phase ORDER BY t.start_date desc))[1] as starting_phase
        FROM annotations a, applications apps
        LEFT JOIN patent_to_trial ptt on ptt.publication_number = apps.publication_number
        LEFT JOIN trials t on t.nct_id = ptt.nct_id
        where a.publication_number = apps.publication_number
        and domain in ('diseases', 'mechanisms', 'biologics', 'compounds')
        group by apps.publication_number
        order by apps.publication_number desc
        limit 10000
    """
    )

    df = pl.from_dicts(patents, infer_schema_length=None)

    df = (
        df.with_columns(
            *[
                df.select(
                    pl.struct(["terms", "domains"])
                    .map_elements(lambda rec: filter_terms_by_domain(rec, domains))  # type: ignore
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
            (pl.col("conditions").list.len() > 0)
            & (pl.col("interventions").list.len() > 0)
        )
    )

    patents = df.to_dicts()

    predictions = predict(patents)

    client.create_and_insert(
        predictions,  # type: ignore
        "patent_clindev_predictions",
        transform=lambda batch, _: transform_predictions(batch),
    )


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
