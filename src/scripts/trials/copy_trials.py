"""
Utils for copying approvals data
"""
import asyncio
import re
import sys
import logging
from pydash import omit
from prisma.enums import BiomedicalEntityType, Source
from prisma.models import (
    Indicatable,
    Intervenable,
    Ownable,
    Trial,
    TrialOutcome,
)
from prisma.types import TrialCreateWithoutRelationsInput

from clients.low_level.postgres import PsqlDatabaseClient
from constants.core import ETL_BASE_DATABASE_URL
from constants.patterns.intervention import DOSAGE_UOM_RE
from data.domain.biomedical import (
    remove_trailing_leading,
    REMOVAL_WORDS_POST as REMOVAL_WORDS,
)
from data.etl.biomedical_entity import BiomedicalEntityEtl
from data.etl.document import DocumentEtl
from data.etl.types import RelationConnectInfo, RelationIdFieldMap
from data.domain.trials import extract_max_timeframe
from scripts.trials.constants import CONTROL_TERMS
from utils.re import get_or_re

from .enums import (
    ComparisonTypeParser,
    HypothesisTypeParser,
    InterventionTypeParser,
    TerminationReasonParser,
    TrialDesignParser,
    TrialMaskingParser,
    TrialPhaseParser,
    TrialPurposeParser,
    TrialRandomizationParser,
    TrialRecord,
    TrialStatusParser,
    calc_duration,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


SOURCE_DB = f"{ETL_BASE_DATABASE_URL}/aact"


def get_source_fields() -> list[str]:
    single_fields = {
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

    multi_fields = {
        "design_groups.group_type": "arm_types",
        "interventions.name": "interventions",  # needed for determining comparison type
        "interventions.intervention_type": "intervention_types",  # needed to determine overall intervention type
        "outcomes.time_frame": "time_frames",  # needed for max_timeframe calc
    }

    single_fields_sql = [f"max({f}) as {new_f}" for f, new_f in single_fields.items()]
    multi_fields_sql = [
        f"array_remove(array_agg(distinct {f}), NULL) as {new_f}"
        for f, new_f in multi_fields.items()
    ]

    return (
        single_fields_sql
        + multi_fields_sql
        + [
            "JSON_AGG(outcomes.*) as outcomes",
            """
            array_remove(array_cat(
                array_agg(distinct lower(conditions.name)),
                array_agg(distinct lower(mesh_conditions.mesh_term))
            ), NULL) as indications
            """,  # needed for search index
        ]
    )


class TrialEtl(DocumentEtl):
    def __init__(self, document_type: str):
        self.document_type = document_type

    @staticmethod
    def get_source_sql(fields: list[str] = get_source_fields()) -> str:
        source_sql = f"""
            select {", ".join(fields)}
            from designs, studies
            JOIN conditions on conditions.nct_id = studies.nct_id
            LEFT JOIN browse_interventions as mesh_interventions on mesh_interventions.nct_id = studies.nct_id
                AND mesh_interventions.mesh_type = 'mesh-list'
            JOIN interventions on interventions.nct_id = studies.nct_id
                AND intervention_type in (
                    'Biological', 'Combination Product', 'Drug',
                    'Dietary Supplement', 'Genetic', 'Other', 'Procedure'
                )
                AND interventions.name is not null
                AND interventions.name ~* '{get_or_re(CONTROL_TERMS)}'
            LEFT JOIN design_groups on design_groups.nct_id = studies.nct_id
            LEFT JOIN browse_conditions as mesh_conditions on mesh_conditions.nct_id = studies.nct_id
                AND mesh_conditions.mesh_type='mesh-list'
            LEFT JOIN outcomes on outcomes.nct_id = studies.nct_id
                AND outcomes.outcome_type = 'Primary'
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
        return source_sql

    async def copy_indications(self):
        """
        Create indication records
        """
        source_sql = f"""
            select distinct name as indication from (
                SELECT DISTINCT lower(mesh_term) as name FROM browse_conditions
                    WHERE mesh_type = 'mesh-list' AND mesh_term is not null
                UNION ALL
                SELECT DISTINCT lower(name) FROM conditions WHERE name is not null
            ) s
        """
        records = await PsqlDatabaseClient(SOURCE_DB).select(query=source_sql)

        insert_map = {
            ir["indication"]: {
                "synonyms": [ir["indication"]],
                "default_type": BiomedicalEntityType.DISEASE,
            }
            for ir in records
        }

        terms = list(insert_map.keys())

        await BiomedicalEntityEtl(
            "CandidateSelector",
            relation_id_field_map=RelationIdFieldMap(
                synonyms=RelationConnectInfo(
                    source_field="synonyms", dest_field="term", input_type="create"
                ),
            ),
            non_canonical_source=Source.CTGOV,
        ).create_records(terms, source_map=insert_map)

    async def copy_interventions(self):
        source_sql = f"""
            select distinct name as intervention from (
                SELECT DISTINCT lower(mesh_term) as name FROM browse_interventions
                    WHERE mesh_type = 'mesh-list' AND mesh_term is not null
                UNION ALL
                SELECT DISTINCT lower(name) FROM interventions
                    WHERE name is not null AND name ~* '{get_or_re(CONTROL_TERMS)}'
            ) s
        """
        records = await PsqlDatabaseClient(SOURCE_DB).select(query=source_sql)

        insert_map = {
            ir["intervention"]: {
                "synonyms": [ir["intervention"]],
                "default_type": BiomedicalEntityType.PHARMACOLOGICAL,  # CONTROL?
            }
            for ir in records
        }

        terms = list(insert_map.keys())

        await BiomedicalEntityEtl(
            "CandidateSelector",
            relation_id_field_map=RelationIdFieldMap(
                synonyms=RelationConnectInfo(
                    source_field="synonyms", dest_field="term", input_type="create"
                ),
            ),
            additional_cleaners=[
                lambda terms: remove_trailing_leading(terms, REMOVAL_WORDS),
                # remove dosage uoms
                lambda terms: [re.sub(DOSAGE_UOM_RE, "", t).strip() for t in terms],
            ],
            non_canonical_source=Source.CTGOV,
        ).create_records(terms, source_map=insert_map)

    @staticmethod
    def transform(record: dict) -> TrialCreateWithoutRelationsInput:
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
                    "indications",
                    "interventions",
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
                "randomization": TrialRandomizationParser.find(
                    trial.randomization, design
                ),
                "text_for_search": f"{trial.title} {trial.sponsor}",
                "status": TrialStatusParser.find(trial.status),
                "termination_reason": TerminationReasonParser.find(
                    trial.termination_description
                ),
                "url": f"https://clinicaltrials.gov/study/{trial.id}",
            },
        )

    async def copy_documents(self):
        """
        Copy data from Postgres (drugcentral) to Postgres (patents)
        """

        async def handle_batch(rows: list[dict]) -> bool:
            # create main trial records
            await Trial.prisma().create_many(
                data=[TrialEtl.transform(t) for t in rows],
                skip_duplicates=True,
            )

            # create owner records (aka sponsors)
            await Ownable.prisma().create_many(
                data=[
                    {
                        "name": (t["sponsor"] or "unknown").lower(),
                        "canonical_name": (t["sponsor"] or "unknown").lower(),
                        "instance_rollup": (t["sponsor"] or "unknown").lower(),
                        "is_primary": True,
                        "trial_id": t["id"],
                    }
                    for t in rows
                ],
                skip_duplicates=True,
            )

            # create "indicatable" records, those that map approval to a canonical indication
            await Indicatable.prisma().create_many(
                data=[
                    {
                        "name": i.lower(),
                        "canonical_name": i.lower(),  # overwritten later
                        "instance_rollup": i.lower(),  # overwritten later
                        "trial_id": t["id"],
                    }
                    for t in rows
                    for i in t["indications"]
                ],
                skip_duplicates=True,
            )

            # create "intervenable" records, those that map approval to a canonical intervention
            await Intervenable.prisma().create_many(
                data=[
                    {
                        "name": i.lower(),
                        "canonical_name": i.lower(),  # overwritten later
                        "instance_rollup": i.lower(),  # overwritten later
                        "is_primary": True,
                        "trial_id": t["id"],
                    }
                    for t in rows
                    for i in t["interventions"]
                ],
                skip_duplicates=True,
            )

            # create "outcome" records
            await TrialOutcome.prisma().create_many(
                data=[
                    {
                        "name": o["title"],
                        # "hypothesis_type": o["hypothesis_type"],
                        "timeframe": o["time_frame"],
                        "trial_id": t["id"],
                    }
                    for t in rows
                    for o in t["outcomes"]
                    if o is not None
                ],
                skip_duplicates=True,
            )

            return True

        await PsqlDatabaseClient(SOURCE_DB).execute_query(
            query=self.get_source_sql(),
            batch_size=1000,
            handle_result_batch=handle_batch,  # type: ignore
        )


def main():
    asyncio.run(TrialEtl(document_type="trial").copy_all())


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m scripts.trials.copy_trials
            Copies ctgov to patents
        """
        )
        sys.exit()

    main()
