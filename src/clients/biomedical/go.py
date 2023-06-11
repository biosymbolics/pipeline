"""
Clients for Gene Ontology (GO)
"""
import requests
from pathlib import Path
import json

path = Path(__file__).parent.parent

QUICK_GO_URL = f"https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/{id}/ancestors?relations=is_a%2Cpart_of%2Coccurs_in%2Cregulates"

ancestor_map: dict[str, list[str]] = {}


def fetch_ancestors(go_id: str) -> list[str]:
    """
    Fetch ancestors for a GO id

    Args:
        go_id (str): GO id
    """
    if not go_id.startswith("GO"):
        raise ValueError(f"Invalid GO id: {go_id}")

    # check cache
    if go_id in ancestor_map:
        return ancestor_map[go_id]

    r = requests.get(QUICK_GO_URL, headers={"Accept": "application/json"})

    try:
        results = json.loads(r.text).get("results")[0].get("ancestors")
        ancestors = results
    except Exception as ex:
        raise Exception(f"Could not fetch ancestors for %s: %s", go_id, ex)

    # add to cache
    ancestor_map[go_id] = ancestors or []
    return ancestors
