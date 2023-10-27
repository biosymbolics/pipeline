"""
Utils for copying approvals data
"""
from functools import reduce
import re
import sys
import logging
from typing import Callable, Sequence
from pydash import flatten

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from core.ner import NerTagger
from constants.core import BASE_DATABASE_URL
from core.ner.cleaning import RE_FLAGS
from data.common.biomedical import (
    remove_trailing_leading,
    REMOVAL_WORDS_POST as REMOVAL_WORDS,
)
from typings.trials import TrialSummary, dict_to_trial_summary
from utils.list import dedup

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SINGLE_FIELDS = {
    "studies.nct_id": "nct_id",
    "studies.acronym": "acronym",
    "studies.brief_title": "title",
    "studies.enrollment": "enrollment",  # Actual or est, by enrollment_type
    "studies.last_update_posted_date": "last_updated_date",
    "studies.overall_status": "status",  # vs last_known_status
    "studies.phase": "phase",
    "studies.primary_completion_date": "end_date",  # Actual or est.
    "studies.source": "sponsor",  # lead sponsor
    "studies.start_date": "start_date",
    "studies.why_stopped": "why_stopped",
    "designs.allocation": "randomization",  # Randomized, Non-Randomized, n/a
    "designs.intervention_model": "design",  # Single Group Assignment, Crossover Assignment, etc.
    "designs.primary_purpose": "purpose",  # Treatment, Prevention, Diagnostic, Supportive Care, Screening, Health Services Research, Basic Science, Device Feasibility
    "designs.masking": "masking",  # None (Open Label), Single (Outcomes Assessor), Double (Participant, Outcomes Assessor), Triple (Participant, Care Provider, Investigator), Quadruple (Participant, Care Provider, Investigator, Outcomes Assessor)
    "drop_withdrawals.dropout_count": "dropout_count",
    "drop_withdrawals.reasons": "dropout_reasons",
    "outcome_analyses.non_inferiority_types": "hypothesis_types",
}

MULTI_FIELDS = {
    "conditions.name": "conditions",
    "mesh_conditions.mesh_term": "mesh_conditions",
    "design_groups.group_type": "arm_types",
    "interventions.name": "interventions",
    "interventions.intervention_type": "intervention_types",
    "outcomes.title": "primary_outcomes",
    "outcomes.time_frame": "time_frames",
}

SINGLE_FIELDS_SQL = [f"max({f}) as {new_f}" for f, new_f in SINGLE_FIELDS.items()]
MULI_FIELDS_SQL = [
    f"array_remove(array_agg(distinct {f}), NULL) as {new_f}"
    for f, new_f in MULTI_FIELDS.items()
]

FIELDS = SINGLE_FIELDS_SQL + MULI_FIELDS_SQL


def is_control(intervention_str: str) -> bool:
    return (
        re.match(
            r".*\b(?:placebo|standard (?:of )?care|sham|standard treatment)s?\b.*",
            intervention_str,
            flags=RE_FLAGS,
        )
        is None
    )


def is_intervention(intervention_str: str) -> bool:
    return not is_control(intervention_str)


def transform_ct_records(
    ctgov_records: Sequence[dict], tagger: NerTagger
) -> Sequence[TrialSummary]:
    """
    Transform ctgov records
    Slow due to intervention mapping!!

    - normalizes/extracts intervention names
    - normalizes status etc.

    good check:
    select intervention, count(*) from trials, unnest(interventions) intervention group by intervention order by count(*) desc;
    """

    intervention_sets = [rec["interventions"] for rec in ctgov_records]

    cleaners: list[Callable[[list[str]], list[str]]] = [
        lambda interventions: dedup(interventions),
        lambda interventions: list(filter(is_intervention, interventions)),
    ]
    interventions = reduce(
        lambda x, cleaner: cleaner(x), cleaners, flatten(intervention_sets)
    )
    logger.info(
        "Extracting intervention names for %s strings (e.g. %s)",
        len(interventions),
        interventions[0:10],
    )
    norm_map = tagger.extract_string_map(interventions)

    def normalize_interventions(interventions: list[str]):
        return flatten([norm_map.get(i) or i for i in interventions])

    return [
        dict_to_trial_summary(
            {**rec, "interventions": normalize_interventions(rec["interventions"])}
        )
        for rec in ctgov_records
    ]

    # return [dict_to_trial_summary(rec) for rec in ctgov_records]


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
        '' as intervention_type,
        -1 as max_timeframe,
        '' as normalized_sponsor,
        '' as sponsor_type,
        '' as termination_reason
        from designs, studies
        JOIN conditions on conditions.nct_id = studies.nct_id
        JOIN interventions on interventions.nct_id = studies.nct_id
            AND intervention_type in (
                'Biological', 'Combination Product', 'Drug',
                'Dietary Supplement', 'Genetic', 'Other', 'Procedure'
            )
        LEFT JOIN design_groups on design_groups.nct_id = studies.nct_id
        LEFT JOIN browse_conditions as mesh_conditions on mesh_conditions.nct_id = studies.nct_id AND mesh_type='mesh-list'
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

    tagger = NerTagger(
        entity_types=frozenset(["compounds", "mechanisms"]),
        link=True,
        additional_cleaners=[
            lambda terms: remove_trailing_leading(terms, REMOVAL_WORDS)
        ],
    )
    trial_db = f"{BASE_DATABASE_URL}/aact"
    PsqlDatabaseClient(trial_db).truncate_table("trials")

    PsqlDatabaseClient.copy_between_db(
        trial_db,
        source_sql,
        f"{BASE_DATABASE_URL}/patents",
        "trials",
        transform=lambda records: transform_ct_records(records, tagger),  # type: ignore
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

    NOTE: we're currently doing some post-hoc term adjustments on annotations,
    and this must be run before.
    """
    client = PsqlDatabaseClient()
    att_query = """
        select a.publication_number, nct_id
        from trials t,
        aggregated_annotations a,
        applications p
        where p.publication_number=a.publication_number
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


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.ctgov.copy_ctgov
            Copies ctgov to patents
        """
        )
        sys.exit()

    copy_ctgov()
