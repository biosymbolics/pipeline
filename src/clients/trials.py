import time
import logging
from typing import Sequence

from clients.low_level.postgres import PsqlDatabaseClient
from typings.trials import TrialSummary, is_trial_summary_list


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fetch_trials(status: str, limit: int = 2000) -> Sequence[TrialSummary]:
    """
    Fetch all trial summaries by status

    Currently specific to the ClinDev model

    Args:
        status (str): Trial status
        limit (int, optional): Number of trials to fetch. Defaults to 2000.
    """
    start = time.time()

    query = f"""
        SELECT *
        FROM trials
        WHERE status=%s
        AND duration > 0
        AND purpose = 'TREATMENT'
        AND array_length(conditions, 1) > 0
        AND array_length(mesh_conditions, 1) > 0
        AND mesh_conditions[1] is not null
        AND array_length(interventions, 1) > 0
        AND comparison_type not in ('UNKNOWN', 'OTHER')
        AND intervention_type='PHARMACOLOGICAL'
        AND design not in ('UNKNOWN', 'FACTORIAL', 'SEQUENTIAL')
        AND randomization not in ('UNKNOWN') -- rare
        AND masking not in ('UNKNOWN')
        AND sponsor_type in ('INDUSTRY', 'INDUSTRY_LARGE', 'OTHER')
        AND enrollment is not null
        ORDER BY start_date DESC
        limit {limit}
    """
    trials = PsqlDatabaseClient().select(query, [status])

    if len(trials) == 0:
        raise ValueError(f"No trials found for status {status}")

    if not is_trial_summary_list(trials):
        raise ValueError(f"Trials are not in summary format: {trials[0:10]}")

    logger.info(
        "Trial fetch for status %s took %s seconds",
        status,
        round(time.time() - start, 2),
    )

    return trials
