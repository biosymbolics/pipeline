"""
ChEMBL client
"""
import requests

from .types import ChemblMolecule, ChemblResponse

CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"


def fetch_by_synonym(synonym: str) -> list[ChemblMolecule]:
    """
    Fetch molecule data by synonym

    Args:
        synonym (str): synonym to search for

    Returns:
        list[ChemblMolecule]: list of molecules
    """
    # search for the synonym
    response = requests.get(f"{CHEMBL_BASE_URL}/molecule/search.json?q={synonym}")
    response.raise_for_status()
    data: ChemblResponse = response.json()

    # retrieve the specific molecule data by ChEMBL ID
    if data["molecules"]:
        return data["molecules"]
    else:
        raise ValueError(f"No molecule found with the synonym {synonym}")


def fetch_by_id(chembl_id: str) -> ChemblMolecule:
    """
    Fetch molecule data by ChEMBL ID

    Args:
        chembl_id (str): ChEMBL ID

    Returns:
        ChemblMolecule: molecule data
    """
    response = requests.get(f"{CHEMBL_BASE_URL}/molecule/{chembl_id}?format=json")
    response.raise_for_status()
    return response.json()
