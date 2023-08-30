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
        select {", ".join(FIELDS)}
        from studies, interventions
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
        select
            (array_agg(studies.nct_id))[1] as nct_id, (array_agg(studies.source))[1] as sponsor,
            (array_agg(studies.start_date))[1] as start_date, (array_agg(studies.completion_date))[0] as end_date,
            array_agg(interventions.name) as intervention_names
        from studies, interventions
        where studies.nct_id = interventions.nct_id
        AND study_type = 'Interventional'
        AND interventions.intervention_type = 'Drug'
        AND interventions.name <> 'Placebo'
        group by studies.nct_id
"""
