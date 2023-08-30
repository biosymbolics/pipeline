"""
Utils for copying approvals data
"""
import sys

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import BASE_DATABASE_URL

SINGLE_FIELDS = {
    "studies.nct_id": "nct_id",
    "studies.source": "sponsor",  # lead sponsor
    "studies.start_date": "start_date",
    "studies.completion_date": "end_date",  # Actual or est.
    "studies.last_update_posted_date": "last_updated_date",
    "studies.target_duration": "duration",
    "studies.why_stopped": "termination_reason",
    "studies.brief_title": "title",
    "studies.phase": "phase",
    "studies.enrollment": "enrollment",  # Actual or est, by enrollment_type
    "studies.overall_status": "status",  # vs last_known_status
    "studies.acronym": "acronym",
}

MULTI_FIELDS = {
    "interventions.name": "intervention_names",
}

SINGLE_FIELDS_SQL = [
    f"(array_agg({f}))[1] as {new_f}" for f, new_f in SINGLE_FIELDS.items()
]
MULI_FIELDS_SQL = [f"array_agg({f}) as {new_f}" for f, new_f in MULTI_FIELDS.items()]
FIELDS = SINGLE_FIELDS_SQL + MULI_FIELDS_SQL


def copy_trials():
    """
    Copy patent clinical trials from ctgov to patents
    """

    source_sql = f"""
        select {", ".join(FIELDS)},
        from
        studies,
        interventions
        where studies.nct_id = interventions.nct_id
        AND study_type = 'Interventional'
        AND interventions.intervention_type = 'Drug'
        AND interventions.name <> 'Placebo'
        group by studies.nct_id
    """
    source_db = f"{BASE_DATABASE_URL}/aact"
    dest_db = f"{BASE_DATABASE_URL}/patents"
    dest_table_name = "trials"
    PsqlDatabaseClient.copy_between_db(
        source_db=source_db,
        source_sql=source_sql,
        dest_db=dest_db,
        dest_table_name=dest_table_name,
    )


def main():
    """
    Copy data from ctgov to patents
    """
    copy_trials()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 copy_psql.py\nCopies ctgov to patents")
        sys.exit()

    main()


"""
select * from trials t, aggregated_annotations anns where t.sponsor=any(anns.terms) and anns.terms && lower(t.intervention_names::text[]) limit 100;


select
LOWER(case when syn_map.term is null then intervention else syn_map.term end) as intervention_name,
t.nct_id, t.sponsor, anns.publication_number
from
trials t,
annotations anns,
aggregated_annotations aaggs,
unnest(t.intervention_names) as intervention
LEFT JOIN synonym_map syn_map on syn_map.synonym = lower(intervention)
where
anns.publication_number = aaggs.publication_number
and t.sponsor=anns.term and anns.domain='assignees'
and LOWER(case when syn_map.term is null then intervention else syn_map.term end) = any(aaggs.terms)
;


select
LOWER(case when syn_map.term is null then intervention else syn_map.term end) as intervention_name,
t.nct_id, t.sponsor, aaggs.publication_number
from
trials t,
aggregated_annotations aaggs,
unnest(t.intervention_names) as intervention
LEFT JOIN synonym_map syn_map on syn_map.synonym = lower(intervention)
where
LOWER(case when syn_map.term is null then intervention else syn_map.term end) = any(aaggs.terms)
;
"""
