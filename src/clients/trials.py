import time
import logging

from clients.low_level.postgres import PsqlDatabaseClient
from typings.trials import TrialSummary, is_trial_summary_list


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fetch_trials(status: str, limit: int = 2000) -> list[TrialSummary]:
    """
    Fetch all trial summaries by status

    Args:
        status (str): Trial status
    """
    start = time.time()

    query = f"""
        SELECT *
        FROM trials
        WHERE status=%s
        AND array_length(conditions, 1) > 0
        AND array_length(interventions, 1) > 0
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
