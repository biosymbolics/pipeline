"""
Utils for copying approvals data
"""
import sys
import logging

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from core.ner.ner import NerTagger
from constants.core import BASE_DATABASE_URL
from utils.list import dedup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SINGLE_FIELDS = {
    "studies.nct_id": "nct_id",
    "studies.source": "sponsor",  # lead sponsor
    "studies.start_date": "start_date",
    "studies.completion_date": "end_date",  # Actual or est.
    "studies.last_update_posted_date": "last_updated_date",
    "studies.why_stopped": "why_stopped",
    "studies.brief_title": "title",
    "studies.phase": "phase",
    "studies.enrollment": "enrollment",  # Actual or est, by enrollment_type
    "studies.overall_status": "status",  # vs last_known_status
    "studies.acronym": "acronym",
}

MULTI_FIELDS = {
    "interventions.name": "interventions",
    "conditions.name": "conditions",
}

SINGLE_FIELDS_SQL = [
    f"(array_agg({f}))[1] as {new_f}" for f, new_f in SINGLE_FIELDS.items()
]
MULI_FIELDS_SQL = [f"array_agg({f}) as {new_f}" for f, new_f in MULTI_FIELDS.items()]
FIELDS = SINGLE_FIELDS_SQL + MULI_FIELDS_SQL


def transform_ct_records(ctgov_records, tagger: NerTagger):
    """
    Transform ctgov records

    - normalizes/extracts intervention names

    TODO:
    - status
    - ??
    """

    intervention_sets: list[list[str]] = [rec["interventions"] for rec in ctgov_records]
    logger.info("Extracting intervention names for %s (...)", intervention_sets[0:10])
    normalized = tagger.extract_strings([dedup(i_set) for i_set in intervention_sets])
    return [
        {**rec, "interventions": normalized}
        for rec, normalized in zip(ctgov_records, normalized)
    ]


def ingest_trials():
    """
    Copy patent clinical trials from ctgov to patents
    """

    source_sql = f"""
        select {", ".join(FIELDS)}
        from studies, interventions, conditions
        where studies.nct_id = interventions.nct_id
        AND studies.nct_id = conditions.nct_id
        AND study_type = 'Interventional'
        AND interventions.intervention_type = 'Drug'
        AND interventions.name not in ('Placebo', 'placebo')
        group by studies.nct_id
    """

    # TODO!
    # update trials set normalized_sponsor=sm.term from
    # synonym_map sm where sm.synonym = lower(sponsor)

    tagger = NerTagger(entity_types=frozenset(["compounds", "mechanisms"]))
    trial_db = f"{BASE_DATABASE_URL}/aact"
    PsqlDatabaseClient(trial_db).truncate_table("trials")

    PsqlDatabaseClient.copy_between_db(
        trial_db,
        source_sql,
        f"{BASE_DATABASE_URL}/patents",
        "trials",
        transform=lambda records: transform_ct_records(records, tagger),
    )

    # TODO create index index_trials_interventions ON trials using gin(interventions);


def main():
    """
    Copy data from ctgov to patents
    """
    ingest_trials()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 -m scripts.ctgov.copy_ctgov\nCopies ctgov to patents")
        sys.exit()

    main()


"""

select apps.publication_number, apps.title as patent, apps.priority_date,
apps.assignees as patent_owner, t.phase as phase, t.interventions
from
trials t,
aggregated_annotations a,
applications apps
where apps.publication_number=a.publication_number
AND t.normalized_sponsor = any(a.terms)
AND t.interventions && a.terms
order by apps.priority_date desc
limit 100;
"""
