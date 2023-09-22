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
from typings.trials import get_trial_summary

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SINGLE_FIELDS = {
    "studies.nct_id": "nct_id",
    "studies.source": "sponsor",  # lead sponsor
    "studies.start_date": "start_date",
    "studies.brief_title": "title",
    "studies.completion_date": "end_date",  # Actual or est.
    "enrollment_type": "enrollment_type",
    "studies.last_update_posted_date": "last_updated_date",
    "studies.why_stopped": "why_stopped",
    "studies.phase": "phase",
    "studies.enrollment": "enrollment",  # Actual or est, by enrollment_type
    "studies.overall_status": "status",  # vs last_known_status
    "studies.acronym": "acronym",
    "designs.intervention_model": "design",  # Single Group Assignment, Crossover Assignment, etc.
    "designs.primary_purpose": "purpose",  # Treatment, Prevention, Diagnostic, Supportive Care, Screening, Health Services Research, Basic Science, Device Feasibility
    "designs.masking": "masking",  # None (Open Label), Single (Outcomes Assessor), Double (Participant, Outcomes Assessor), Triple (Participant, Care Provider, Investigator), Quadruple (Participant, Care Provider, Investigator, Outcomes Assessor)
    "designs.allocation": "randomization",  # Randomized, Non-Randomized, n/a
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


def transform_ct_records(ctgov_records: list[dict], tagger: NerTagger):
    """
    Transform ctgov records
    Slow due to intervention mapping!!

    - normalizes/extracts intervention names
    - normalizes status etc.
    """

    # intervention_sets: list[list[str]] = [rec["interventions"] for rec in ctgov_records]
    # logger.info("Extracting intervention names for %s (...)", intervention_sets[0:10])
    # normalized = tagger.extract_strings([dedup(i_set) for i_set in intervention_sets])
    return [{**get_trial_summary(rec)} for rec in ctgov_records]


def ingest_trials():
    """
    Copy patent clinical trials from ctgov to patents
    TODO: use sponsors table to get agency_class
    """

    source_sql = f"""
        select {", ".join(FIELDS)},
        0 as duration,
        '' as normalized_sponsor,
        '' as sponsor_type,
        '' as termination_reason
        from studies, interventions, conditions, designs
        where studies.nct_id = interventions.nct_id
        AND studies.nct_id = conditions.nct_id
        AND study_type = 'Interventional'
        AND designs.nct_id = studies.nct_id
        AND interventions.intervention_type = 'Drug'
        AND not interventions.name ~* '(?:saline|placebo)'
        group by studies.nct_id
    """

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
    # TODO: alter table trials alter column interventions set data type text[];

    client = PsqlDatabaseClient()

    # NOTE: this is a flaw. synonym_map is populated with trial sponsors,
    # so this will only work if we're loading trials for a subsequent time.
    client.execute_query(
        """
        update trials set normalized_sponsor=sm.term from
        synonym_map sm where sm.synonym = lower(sponsor)
        """
    )
    client.create_indices(
        [
            {
                "table": "trials",
                "column": "nct_id",
            },
            {
                "table": "trials",
                "column": "interventions",
                "is_gin": True,
            },
            {
                "table": "trials",
                "column": "normalized_sponsor",
            },
        ]
    )


def create_patent_to_trial():
    """
    Create table that maps patent applications to trials
    """
    client = PsqlDatabaseClient()
    att_query = """
        select a.publication_number, nct_id
        from trials t,
        aggregated_annotations a,
        applications p
        where a.publication_number=a.publication_number
        AND p.publication_number=a.publication_number
        AND t.normalized_sponsor = any(a.terms) -- sponsor match
        AND t.interventions && a.terms -- intervention match
        AND t.start_date >= p.priority_date -- seemingly the trial starts after the patent was filed
    """
    client.create_from_select(att_query, "patent_to_trial")
    client.create_indices(
        [
            {
                "table": "patent_to_trial",
                "column": "publication_number",
            },
            {
                "table": "patent_to_trial",
                "column": "nct_id",
            },
        ]
    )
    client.insert_into_table


def copy_ctgov():
    """
    Copy data from ctgov to patents
    """
    ingest_trials()
    create_patent_to_trial()


def main():
    copy_ctgov()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
              Usage: python3 -m scripts.ctgov.copy_ctgov
              Copies ctgov to patents
        """
        )
        sys.exit()

    main()
