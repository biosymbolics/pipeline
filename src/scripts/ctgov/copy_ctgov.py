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
    "studies.primary_completion_date": "end_date",  # Actual or est.
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
    "drop_withdrawals.reasons": "dropout_reasons",
    "drop_withdrawals.dropout_count": "dropout_count",
    "outcome_analyses.non_inferiority_types": "hypothesis_types",
}

MULTI_FIELDS = {
    "conditions.name": "conditions",
    "mesh_conditions.mesh_term": "mesh_conditions",
    "design_groups.group_type": "arm_types",
    "interventions.name": "interventions",
    "outcomes.title": "primary_outcomes",
}

SINGLE_FIELDS_SQL = [f"max({f}) as {new_f}" for f, new_f in SINGLE_FIELDS.items()]
MULI_FIELDS_SQL = [
    f"array_agg(distinct {f}) as {new_f}" for f, new_f in MULTI_FIELDS.items()
]

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

    # reported_events (event_type == 'serious', adverse_event_term, subjects_affected)
    # subqueries to avoid combinatorial explosion
    source_sql = f"""
        select {", ".join(FIELDS)},
        0 as duration,
        '' as comparison_type,
        '' as hypothesis_type,
        '' as normalized_sponsor,
        '' as sponsor_type,
        '' as termination_reason
        from designs, studies
        JOIN conditions on conditions.nct_id = studies.nct_id
        JOIN interventions on interventions.nct_id = studies.nct_id AND intervention_type = 'Drug'
        LEFT JOIN design_groups on design_groups.nct_id = studies.nct_id
        LEFT JOIN browse_conditions as mesh_conditions on mesh_conditions.nct_id = studies.nct_id
        LEFT JOIN outcomes on outcomes.nct_id = studies.nct_id AND outcomes.outcome_type = 'Primary'
        LEFT JOIN (
            select nct_id, sum(count) as dropout_count, array_agg(distinct reason) as reasons
            from drop_withdrawals
            group by nct_id
        ) drop_withdrawals on drop_withdrawals.nct_id = studies.nct_id
        LEFT JOIN (
            select
            nct_id,
            array_agg(distinct non_inferiority_type) as non_inferiority_types
            from outcome_analyses
            group by nct_id
        ) outcome_analyses on outcome_analyses.nct_id = studies.nct_id
        where study_type = 'Interventional'
        AND designs.nct_id = studies.nct_id
        group by studies.nct_id
    """

    tagger = NerTagger(entity_types=frozenset(["compounds", "mechanisms"]), link=False)
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
    # TODO: this isn't working as expected; maybe some sponsors are missing from synonym_map?
    # e.g. "respirion pharmaceuticals pty ltd" -> "respirion"
    # "british university in egypt" -> "british university"
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
        AND t.interventions::text[] && a.terms -- intervention match
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
