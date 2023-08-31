from typing import TypedDict


PatentTrialRecord = TypedDict(
    "PatentTrialRecord",
    {
        "nct_id": str,
        "acronym": str,
        "brief_title": str,
        "completion_date": str,
        "enrollment": int,
        "interventions": list[str],
        "last_updated_date": str,
        "overall_status": str,
        "phase": str,
        "sponsor": str,
        "start_date": str,
        "title": str,
        "why_stopped": str,
    },
)
