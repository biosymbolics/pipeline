"""
Load script for Trial etl
"""
import asyncio
import re
import sys
import logging
from typing import Sequence
from pydash import omit, uniq
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
from clients.low_level.prisma import prisma_client
from constants.core import ETL_BASE_DATABASE_URL
from constants.patterns.intervention import DOSAGE_UOM_RE
from data.domain.biomedical import (
    remove_trailing_leading,
    REMOVAL_WORDS_POST as REMOVAL_WORDS,
)
from data.etl.types import BiomedicalEntityLoadSpec
from data.domain.trials import extract_max_timeframe
from utils.classes import overrides
from utils.re import get_or_re

from .constants import CONTROL_TERMS
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

from ..base_document import BaseDocumentEtl


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
            """,
        ]
    )


class TrialLoader(BaseDocumentEtl):
    """
    Load trials and associated entities
    """

    def __init__(self, document_type: str):
        self.document_type = document_type

    @staticmethod
    def get_source_sql(fields: list[str] = get_source_fields()) -> str:
        source_sql = f"""
            SELECT {", ".join(fields)}
            FROM designs, studies
            JOIN conditions on conditions.nct_id = studies.nct_id
            LEFT JOIN browse_interventions as mesh_interventions on mesh_interventions.nct_id = studies.nct_id
                AND mesh_interventions.mesh_type = 'mesh-list'
            JOIN interventions on interventions.nct_id = studies.nct_id
                AND intervention_type in (
                    'Biological', 'Combination Product', 'Drug',
                    'Dietary Supplement', 'Genetic', 'Other', 'Procedure'
                )
                AND interventions.name is not null
                AND NOT interventions.name ~* '{get_or_re(CONTROL_TERMS)}'
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
            WHERE study_type = 'Interventional'
            AND designs.nct_id = studies.nct_id
            GROUP BY studies.nct_id
        """
        return source_sql

    @staticmethod
    @overrides(BaseDocumentEtl)
    def entity_specs() -> list[BiomedicalEntityLoadSpec]:
        """
        Specs for creating associated biomedical entities (executed by BiomedicalEntityEtl)
        """

        def get_sql(mesh_table: str, table: str, filters: Sequence[str] = []):
            return f"""
                SELECT DISTINCT name FROM (
                    SELECT DISTINCT LOWER(mesh_term) AS name
                        FROM {mesh_table}
                        WHERE mesh_type = 'mesh-list' AND mesh_term IS NOT null
                    UNION
                    SELECT DISTINCT LOWER(name) FROM {table} WHERE name IS NOT null
                ) s
                {'WHERE ' if filters else ''}
                {' AND '.join(filters)}
            """

        indication_spec = BiomedicalEntityLoadSpec(
            candidate_selector="CandidateSelector",
            database=f"{ETL_BASE_DATABASE_URL}/aact",
            get_source_map=lambda recs: {
                rec["name"]: {
                    "synonyms": [rec["name"]],
                    "default_type": BiomedicalEntityType.DISEASE,
                }
                for rec in recs
            },
            non_canonical_source=Source.CTGOV,
            sql=get_sql("browse_conditions", "conditions"),
        )
        intervention_spec = BiomedicalEntityLoadSpec(
            additional_cleaners=[
                lambda terms: remove_trailing_leading(terms, REMOVAL_WORDS),
                # remove dosage uoms
                lambda terms: [re.sub(DOSAGE_UOM_RE, "", t).strip() for t in terms],
            ],
            candidate_selector="CandidateSelector",
            database=f"{ETL_BASE_DATABASE_URL}/aact",
            get_source_map=lambda recs: {
                rec["name"]: {
                    "synonyms": [rec["name"]],
                    "default_type": BiomedicalEntityType.PHARMACOLOGICAL,  # CONTROL?
                }
                for rec in recs
            },
            non_canonical_source=Source.CTGOV,
            sql=get_sql(
                "browse_interventions",
                "interventions",
                [f"AND NOT interventions.name ~* '{get_or_re(CONTROL_TERMS)}'"],
            ),
        )
        return [indication_spec, intervention_spec]

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
                "status": TrialStatusParser.find(trial.status),
                "termination_reason": TerminationReasonParser.find(
                    trial.termination_description
                ),
                "url": f"https://clinicaltrials.gov/study/{trial.id}",
            },
        )

    @overrides(BaseDocumentEtl)
    async def copy_documents(self):
        """
        Copy data from Postgres (drugcentral) to Postgres (patents)
        """

        client = await prisma_client(600)

        async def handle_batch(rows: list[dict]) -> bool:
            # create main trial records
            await Trial.prisma(client).create_many(
                data=[TrialLoader.transform(t) for t in rows],
                skip_duplicates=True,
            )

            # # create owner records (aka sponsors)
            await Ownable.prisma(client).create_many(
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

            # # create "indicatable" records, those that map approval to a canonical indication
            await Indicatable.prisma(client).create_many(
                data=[
                    {
                        "name": i.lower(),
                        "canonical_name": i.lower(),  # overwritten later
                        "trial_id": t["id"],
                    }
                    for t in rows
                    for i in uniq(t["indications"])
                ],
                skip_duplicates=True,
            )

            # create "intervenable" records, those that map approval to a canonical intervention
            await Intervenable.prisma(client).create_many(
                data=[
                    {
                        "name": i.lower(),
                        "canonical_name": i.lower(),  # overwritten later
                        "is_primary": True,
                        "trial_id": t["id"],
                    }
                    for t in rows
                    for i in uniq(t["interventions"])
                ],
                skip_duplicates=True,
            )

            # create "outcome" records
            await TrialOutcome.prisma(client).create_many(
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
            batch_size=5000,
            handle_result_batch=handle_batch,  # type: ignore
        )


def main():
    asyncio.run(TrialLoader(document_type="trial").copy_all())


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.documents.trial.load
            Copies ctgov to patents
        """
        )
        sys.exit()

    main()
