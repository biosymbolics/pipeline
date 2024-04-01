"""
Load script for Trial etl
"""

import asyncio
import re
import sys
import logging
from typing import Sequence
from pydash import uniq
from prisma.enums import BiomedicalEntityType, Source, TrialStatus
from prisma.models import (
    Indicatable,
    Intervenable,
    Ownable,
    Trial,
    TrialDropoutReason,
    TrialOutcome,
)
from prisma.types import TrialCreateWithoutRelationsInput, TrialDropoutReasonCreateInput

from clients.low_level.postgres import PsqlDatabaseClient
from clients.low_level.prisma import prisma_client
from constants.core import ETL_BASE_DATABASE_URL
from constants.patterns.intervention import DOSAGE_UOM_RE
from constants.documents import TRIAL_WEIGHT_MULTIPLIER
from data.domain.biomedical import (
    remove_trailing_leading,
    REMOVAL_WORDS_POST as REMOVAL_WORDS,
)
from data.etl.types import BiomedicalEntityLoadSpec
from typings import DocType
from utils.classes import overrides
from utils.re import get_or_re

from .constants import CONTROL_TERMS
from .parsers import (
    ComparisonTypeParser,
    DropoutReasonParser,
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
from .parsers import extract_max_timeframe

from ..base_document_etl import BaseDocumentEtl


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
        "brief_summaries.description": "abstract",
        "designs.allocation": "randomization",  # Randomized, Non-Randomized, n/a
        "designs.intervention_model": "design",  # Single Group Assignment, Crossover Assignment, etc.
        "designs.primary_purpose": "purpose",  # Treatment, Prevention, Diagnostic, Supportive Care, Screening, Health Services Research, Basic Science, Device Feasibility
        "designs.masking": "masking",  # None (Open Label), Single (Outcomes Assessor), Double (Participant, Outcomes Assessor), Triple (Participant, Care Provider, Investigator), Quadruple (Participant, Care Provider, Investigator, Outcomes Assessor)
        "outcome_analyses.non_inferiority_types": "hypothesis_types",
        "dropout_count": "dropout_count",
        "time_frames": "time_frames",  # needed for max_timeframe calc
        "interventions": "interventions",
        "intervention_types": "intervention_types",
        "indications": "indications",
        "arm_types": "arm_types",
        "primary_outcomes": "outcomes",
        "dropout_reasons": "dropout_reasons",
    }

    fields_sql = [f"{f} as {new_f}" for f, new_f in single_fields.items()]
    return fields_sql


class TrialLoader(BaseDocumentEtl):
    """
    Load trials and associated entities
    """

    @staticmethod
    def get_source_sql(fields: list[str] = get_source_fields()) -> str:
        source_sql = f"""
            SELECT {", ".join(fields)}
            FROM designs, studies
            LEFT JOIN (
                SELECT
                    nct_id,
                    ARRAY_AGG(group_type) as arm_types
                FROM design_groups
                GROUP BY nct_id
            ) design_groups on design_groups.nct_id = studies.nct_id
            LEFT JOIN (
                SELECT
                    nct_id,
                    ARRAY_AGG(name) as indications
                FROM (
                    SELECT
                        nct_id,
                        ARRAY_AGG(distinct lower(mesh_term)) as names
                    FROM browse_conditions
                    WHERE mesh_type = 'mesh-list'
                    GROUP BY nct_id

                    UNION

                    SELECT
                        nct_id,
                        ARRAY_AGG(distinct lower(name)) as names
                    FROM conditions
                    GROUP BY nct_id
                ) s,
                UNNEST(names) as name
                GROUP BY nct_id
            ) conditions on conditions.nct_id = studies.nct_id
            LEFT JOIN (
                SELECT
                    nct_id,
                    ARRAY_AGG(name) as interventions,
                    ARRAY_AGG(intervention_type) as intervention_types
                FROM (
                    SELECT
                        nct_id,
                        ARRAY_AGG(distinct lower(mesh_term)) as names,
                        ARRAY[]::text[] as intervention_types
                    FROM browse_interventions
                    WHERE mesh_type = 'mesh-list'
                    GROUP BY nct_id

                    UNION

                    SELECT
                        nct_id,
                        ARRAY_AGG(distinct lower(name)) as names,
                        ARRAY_AGG(distinct intervention_type) as intervention_types
                    FROM interventions
                    WHERE intervention_type in (
                        'Biological', 'Combination Product', 'Drug',
                        'Dietary Supplement', 'Genetic', 'Other', 'Procedure'
                    )
                    AND name IS NOT null
                    AND NOT name ~* '{get_or_re(CONTROL_TERMS)}'
                    GROUP BY nct_id
                ) s,
                UNNEST(names) as name,
                UNNEST(intervention_types) as intervention_type
                GROUP BY nct_id
            ) interventions on interventions.nct_id = studies.nct_id
            LEFT JOIN (
                SELECT
                    nct_id,
                    JSON_AGG(outcomes.*) as primary_outcomes,
                    ARRAY_AGG(time_frame) as time_frames
                FROM outcomes
                WHERE outcomes.outcome_type = 'Primary'
                GROUP BY nct_id
            ) outcomes on outcomes.nct_id = studies.nct_id
            LEFT JOIN (
                SELECT
                    nct_id,
                    sum(count) as dropout_count,
                    JSON_AGG(json_build_object('reason', reason, 'count', count)) as dropout_reasons
                FROM (
                    SELECT
                        nct_id,
                        sum(count) as count,
                        reason
                    FROM drop_withdrawals
                    GROUP BY nct_id, reason
                ) s
                GROUP BY nct_id
            ) drop_withdrawals on drop_withdrawals.nct_id = studies.nct_id
            LEFT JOIN (
                SELECT
                    nct_id,
                    ARRAY_AGG(distinct non_inferiority_type) as non_inferiority_types
                FROM outcome_analyses
                GROUP BY nct_id
            ) outcome_analyses on outcome_analyses.nct_id = studies.nct_id
            LEFT JOIN brief_summaries on brief_summaries.nct_id = studies.nct_id
            WHERE study_type = 'Interventional'
            AND designs.nct_id = studies.nct_id
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
            candidate_selector="CompositeCandidateSelector",
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
            candidate_selector="CompositeCandidateSelector",
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
                [f"NOT name ~* '{get_or_re(CONTROL_TERMS)}'"],
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
        status = TrialStatusParser.find(trial.status)

        return TrialCreateWithoutRelationsInput(
            id=trial.id,
            acronym=trial.acronym,
            arm_count=int(trial.arm_count or 0),
            comparison_type=ComparisonTypeParser.find(
                trial["arm_types"] or [], trial["interventions"], design
            ),
            design=design,
            dropout_count=int(trial.dropout_count or 0),
            duration=calc_duration(trial.start_date, trial.end_date),
            end_date=trial.end_date,
            enrollment=int(trial.enrollment or 0),
            hypothesis_type=HypothesisTypeParser.find(trial.hypothesis_types),
            intervention_type=InterventionTypeParser.find(trial.intervention_types),
            investment=(trial.enrollment or 0) * TRIAL_WEIGHT_MULTIPLIER,
            last_updated_date=trial.last_updated_date,
            masking=masking,
            max_timeframe=extract_max_timeframe(trial["time_frames"]),
            phase=TrialPhaseParser.find(trial.phase),
            purpose=TrialPurposeParser.find(trial.purpose).name,
            randomization=TrialRandomizationParser.find(trial.randomization, design),
            start_date=trial.start_date,
            status=status,
            termination_reason=TerminationReasonParser.find(
                trial.termination_description
            ),
            termination_description=trial.termination_description,
            title=trial.title,
            traction=((trial.enrollment or 0) if status == TrialStatus.COMPLETED else 0)
            * TRIAL_WEIGHT_MULTIPLIER,
            url=f"https://clinicaltrials.gov/study/{trial.id}",
        )

    @overrides(BaseDocumentEtl)
    async def delete_all(self):
        """
        Delete all trial records
        """
        client = await prisma_client(600)
        await TrialOutcome.prisma(client).delete_many()
        await TrialDropoutReason.prisma(client).delete_many()
        await Ownable.prisma(client).query_raw(
            "DELETE FROM ownable WHERE trial_id IS NOT NULL"
        )
        await Intervenable.prisma(client).query_raw(
            "DELETE FROM intervenable WHERE trial_id IS NOT NULL"
        )
        await Indicatable.prisma(client).query_raw(
            "DELETE FROM indicatable WHERE trial_id IS NOT NULL"
        )
        await Trial.prisma(client).delete_many()

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

            # create owner records (aka sponsors)
            await Ownable.prisma(client).create_many(
                data=[
                    {
                        "name": (t["sponsor"] or "unknown").lower(),
                        "canonical_name": (t["sponsor"] or "unknown").lower(),
                        "instance_rollup": (t["sponsor"] or "unknown").lower(),
                        "category_rollup": (t["sponsor"] or "unknown").lower(),
                        "is_primary": True,
                        "trial_id": t["id"],
                    }
                    for t in rows
                ],
                skip_duplicates=True,
            )

            # create "indicatable" records, those that map approval to a canonical indication
            await Indicatable.prisma(client).create_many(
                data=[
                    {
                        "name": i.lower(),
                        "canonical_name": i.lower(),  # overwritten later
                        "trial_id": t["id"],
                    }
                    for t in rows
                    for i in uniq(t["indications"] or [])
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
                    for i in uniq(t["interventions"] or [])
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
                    for o in t["outcomes"] or []
                    if o is not None
                ],
                skip_duplicates=True,
            )

            await TrialDropoutReason.prisma(client).create_many(
                data=[
                    TrialDropoutReasonCreateInput(
                        reason=DropoutReasonParser.find(str(dr["reason"])),
                        count=int(dr["count"]),
                        trial_id=t["id"],
                    )
                    for t in rows
                    for dr in t["dropout_reasons"] or []
                    if "reason" in dr and "count" in dr
                ],
                skip_duplicates=True,
            )

            return True

        await PsqlDatabaseClient(self.source_db).execute_query(
            query=self.get_source_sql(),
            batch_size=10000,
            handle_result_batch=handle_batch,  # type: ignore
        )


if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            """
            Usage: python3 -m data.etl.documents.trial.load_trial [--update]
            Copies ctgov to patents
        """
        )
        sys.exit()

    is_update = "--update" in sys.argv

    asyncio.run(
        TrialLoader(document_type=DocType.trial, source_db="aact").copy_all(is_update)
    )
