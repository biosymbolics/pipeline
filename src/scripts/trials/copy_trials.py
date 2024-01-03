"""
Utils for copying approvals data
"""
import asyncio
import datetime
from functools import reduce
import re
import sys
import logging
from typing import Callable, Sequence
from prisma import Prisma
from pydash import compact, flatten, omit
from prisma.models import (
    Indicatable,
    Intervenable,
    Ownable,
    Trial,
    TrialOutcome,
)
from prisma.types import TrialCreateWithoutRelationsInput


from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import ETL_BASE_DATABASE_URL, TRIALS_TABLE
from core.ner.cleaning import RE_FLAGS
from data.domain.biomedical import (
    remove_trailing_leading,
    REMOVAL_WORDS_POST as REMOVAL_WORDS,
)
from data.domain.trials import extract_max_timeframe
from scripts.approvals.copy_approvals import get_preferred_pharmacologic_class

from .enums import (
    ComparisonTypeParser,
    HypothesisTypeParser,
    InterventionTypeParser,
    SponsorTypeParser,
    TerminationReasonParser,
    TrialDesignParser,
    TrialMasking,
    TrialMaskingParser,
    TrialPhaseParser,
    TrialPurposeParser,
    TrialRandomizationParser,
    TrialRecord,
    TrialStatusParser,
    TrialSummary,
    calc_duration,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SINGLE_FIELDS = {
    "studies.nct_id": "id",
    "studies.acronym": "acronym",
    "coalesce(studies.official_title, studies.brief_title)": "title",
    "studies.enrollment": "enrollment",  # Actual or est, by enrollment_type
    "studies.last_update_posted_date::TIMESTAMP": "last_updated_date",
    "studies.overall_status": "status",  # vs last_known_status
    "studies.phase": "phase",
    "studies.number_of_arms": "arm_count",
    "studies.primary_completion_date::TIMESTAMP": "end_date",  # Actual or est.
    "studies.source": "sponsor",  # lead sponsor
    "studies.start_date::TIMESTAMP": "start_date",
    "studies.why_stopped": "termination_description",
    "designs.allocation": "randomization",  # Randomized, Non-Randomized, n/a
    "designs.intervention_model": "design",  # Single Group Assignment, Crossover Assignment, etc.
    "designs.primary_purpose": "purpose",  # Treatment, Prevention, Diagnostic, Supportive Care, Screening, Health Services Research, Basic Science, Device Feasibility
    "designs.masking": "masking",  # None (Open Label), Single (Outcomes Assessor), Double (Participant, Outcomes Assessor), Triple (Participant, Care Provider, Investigator), Quadruple (Participant, Care Provider, Investigator, Outcomes Assessor)
    "drop_withdrawals.dropout_count": "dropout_count",
    "drop_withdrawals.reasons": "dropout_reasons",
    "outcome_analyses.non_inferiority_types": "hypothesis_types",
    # TODO this will break; just a placeholder.
    # update trials set intervention=lower(interventions[0]) where array_length(interventions, 1) > 0 and intervention is null;
    # "COALESCE(mesh_interventions[1].mesh_term, interventions[1].name)": "intervention",  # TODO: there can be many, tho not sure if those are combos or comparators
}

MULTI_FIELDS = {
    "design_groups.group_type": "arm_types",
    "interventions.name": "interventions",
    "interventions.intervention_type": "intervention_types",
    "outcomes.time_frame": "time_frames",  # needed for max_timeframe calc
}

SINGLE_FIELDS_SQL = [f"max({f}) as {new_f}" for f, new_f in SINGLE_FIELDS.items()]
MULI_FIELDS_SQL = [
    f"array_remove(array_agg(distinct {f}), NULL) as {new_f}"
    for f, new_f in MULTI_FIELDS.items()
]

SOURCE_FIELDS = (
    SINGLE_FIELDS_SQL
    + MULI_FIELDS_SQL
    + [
        """
        array_remove(array_cat(
            array_agg(distinct conditions.name),
            array_agg(distinct mesh_conditions.mesh_term)
        ), NULL) as indications
        """,
        """
        JSON_AGG(outcomes.*) as outcomes
        """,
    ]
)


def get_source_sql(fields=SOURCE_FIELDS):
    source_sql = f"""
        select {", ".join(fields)}
        from designs, studies
        JOIN conditions on conditions.nct_id = studies.nct_id
        LEFT JOIN browse_interventions as mesh_interventions on mesh_interventions.nct_id = studies.nct_id AND mesh_interventions.mesh_type = 'mesh-list'
        JOIN interventions on interventions.nct_id = studies.nct_id
            AND intervention_type in (
                'Biological', 'Combination Product', 'Drug',
                'Dietary Supplement', 'Genetic', 'Other', 'Procedure'
            )
        LEFT JOIN design_groups on design_groups.nct_id = studies.nct_id
        LEFT JOIN browse_conditions as mesh_conditions on mesh_conditions.nct_id = studies.nct_id AND mesh_conditions.mesh_type='mesh-list'
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
        limit 100
    """
    return source_sql


SEARCH_FIELDS = {
    "title": "title",
    "acronym": "coalesce(acronym, '')",
    "conditions": "array_to_string(conditions, ' ')",
    "interventions": "array_to_string(interventions, ' ')",
    "mesh_conditions": "array_to_string(mesh_conditions, ' ')",
    "normalized_sponsor": "coalesce(normalized_sponsor, '')",
    "sponsor": "coalesce(sponsor, '')",
    "pharmacologic_class": "coalesce(pharmacologic_class, '')",
}


def is_intervention(intervention_str: str) -> bool:
    def is_control(intervention_str: str) -> bool:
        return (
            re.match(
                r".*\b(?:placebo|sham|best supportive care|standard|usual care|comparator|no treatment|saline solution|conventional|aspirin|control|Tablet Dosage Form|Laboratory Biomarker Analysis|Drug vehicle|pharmacological study|Normal saline|Therapeutic procedure|Quality-of-Life Assessment|Questionnaire Administration|Dosage)s?\b.*",
                intervention_str,
                flags=RE_FLAGS,
            )
            is not None
        )

    return not is_control(intervention_str)


def raw_to_trial_summary(record: dict) -> TrialCreateWithoutRelationsInput:
    """
    Get trial summary from db record

    - formats start and end date
    - calculates duration
    - etc
    """
    trial = TrialRecord(**record)
    design = TrialDesignParser.find_from_record(trial)
    masking = TrialMaskingParser.find(trial.masking)

    return TrialCreateWithoutRelationsInput(
        **{
            **omit(
                trial,
                "hypothesis_types",
                "interventions",
                "indications",
                "intervention_types",
                "outcomes",
                "sponsor",
                "time_frames",
            ),  # type: ignore
            # "blinding": TrialBlinding.find(masking),
            "comparison_type": ComparisonTypeParser.find(
                trial["arm_types"] or [], trial["interventions"], design
            ),
            "design": design,
            "dropout_reasons": trial.dropout_reasons or [],
            "duration": calc_duration(trial["start_date"], trial["end_date"]),  # type: ignore
            "max_timeframe": extract_max_timeframe(trial["time_frames"]),
            "hypothesis_type": HypothesisTypeParser.find(trial["hypothesis_types"]),
            "intervention_type": InterventionTypeParser.find(
                trial["intervention_types"]
            ),
            "masking": masking,
            "phase": TrialPhaseParser.find(trial.phase),
            "purpose": TrialPurposeParser.find(trial.purpose),
            "randomization": TrialRandomizationParser.find(trial.randomization, design),
            "text_for_search": f"{trial.title} {' '.join(trial.interventions)} {' '.join(trial.indications)}",
            "status": TrialStatusParser.find(trial.status),
            "termination_reason": TerminationReasonParser.find(
                trial.termination_description
            ),
            "url": f"https://clinicaltrials.gov/study/{trial.id}",
        },
    )


SOURCE_DB = f"{ETL_BASE_DATABASE_URL}/aact"


async def ingest_trials():
    """
    Copy data from Postgres (drugcentral) to Postgres (patents)
    """

    additional_cleaners = [
        lambda terms: remove_trailing_leading(terms, REMOVAL_WORDS),
        lambda interventions: list(filter(is_intervention, interventions)),
    ]
    source_records = PsqlDatabaseClient(SOURCE_DB).select(query=get_source_sql())
    db = Prisma(auto_register=True)
    await db.connect()

    # create main trial records
    await Trial.prisma().create_many(
        data=[raw_to_trial_summary(t) for t in source_records],
        skip_duplicates=True,
    )

    # create owner records (aka sponsors)
    await Ownable.prisma().create_many(
        data=[
            {
                "name": t["sponsor"] or "unknown",
                "is_primary": True,
                "trial_id": t["id"],
                # SponsorTypeParser.find(trial.sponsor)
            }
            for t in source_records
        ],
        skip_duplicates=True,
    )

    # create "indicatable" records, those that map approval to a canonical indication
    await Indicatable.prisma().create_many(
        data=[
            {"name": i.lower(), "trial_id": t["id"]}
            for t in source_records
            for i in t["indications"]
        ],
        skip_duplicates=True,
    )

    # create "intervenable" records, those that map approval to a canonical intervention
    await Intervenable.prisma().create_many(
        data=[
            {
                "name": i.lower(),
                "instance_rollup": i.lower(),
                "is_primary": True,
                "trial_id": t["id"],
            }
            for t in source_records
            for i in t["interventions"]
        ],
        skip_duplicates=True,
    )

    await TrialOutcome.prisma().create_many(
        data=[
            {
                "name": o["title"],
                # "hypothesis_type": o["hypothesis_type"],
                "timeframe": o["time_frame"],
                "trial_id": t["id"],
            }
            for t in source_records
            for o in t["outcomes"]
            if o is not None
        ],
        skip_duplicates=True,
    )

    await db.disconnect()

    # create search index (unsupported by Prisma)
    raw_client = PsqlDatabaseClient()
    raw_client.execute_query(
        f"""
        UPDATE {TRIALS_TABLE} SET search = to_tsvector('english', text_for_search)
        """,
    )
    raw_client.create_indices(
        [
            {
                "table": TRIALS_TABLE,
                "column": "search",
                "is_gin": True,
            },
        ]
    )


async def copy_ctgov():
    """
    Copy data from ctgov to patents
    """
    await ingest_trials()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.trials.copy_trials
            Copies ctgov to patents
        """
        )
        sys.exit()

    asyncio.run(copy_ctgov())
