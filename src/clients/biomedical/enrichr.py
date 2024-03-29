from typing import Sequence, cast
import requests
import polars as pl
from typing_extensions import TypedDict
import json

ENRICHR_URL = "https://maayanlab.cloud/Enrichr"
DEFAULT_GENE_LIBRARY = "REACTOME_2022"

EnrichrEntity = TypedDict(
    "EnrichrEntity",
    {
        "rank": float,
        "term": str,
        "p-value": float,
        "z-score": float,
        "combined score": float,
        "overlapping genes": str,
        "adjusted p-value": float,
        "old p-value": float,
        "old adjusted p-value": float,
    },
)


def _format_gene_list(genes: Sequence[str]) -> str:
    """
    Format genes how Enrichr requires (as a string; one per line)

    Args:
        genes (list[str]): List of genes
    """
    return "\n".join(genes)


def _generate_list(genes: Sequence[str]) -> str:
    """
    Generate enrichr list (which is later retrieved)

    Args:
        genes (list[str]): List of genes
    """
    response: requests.Response = requests.post(
        f"{ENRICHR_URL}/addList",
        files={"list": (None, _format_gene_list(genes))},
    )

    if not response.ok:
        raise Exception(f"Error analyzing gene list {response.reason}")

    list_id = json.loads(response.text)["userListId"]

    return list_id


def _retrieve_list(list_id: str, gene_library: str) -> list[EnrichrEntity]:
    """
    Retrieve enrichr gene list

    Example result:
    ```json
    [{'rank': 1.0, 'term': 'focal adhesion', 'p-value': 4.4414325372929336e-05, 'z-score': 10.442622950819672,
    'combined score': 104.65542938701697, 'overlapping genes': ['COL4A2', 'COL4A1', 'CAV1', 'COL5A2', 'FLNC', 'EGFR'],
    'adjusted p-value': 0.001465672737306668, 'old p-value': 0.0, 'old adjusted p-value': 0.0}, ...]
    ```
    """
    query_params = f"userListId={list_id}&backgroundType={gene_library}"
    response = requests.get(f"{ENRICHR_URL}/enrich?{query_params}")

    response.raise_for_status()

    enriched_df = pl.DataFrame(
        data=json.loads(response.text).get(gene_library),
        schema={
            # order matters
            "rank": pl.Float64,
            "term": str,
            "p-value": pl.Float64,
            "z-score": pl.Float64,
            "combined score": pl.Float64,
            "overlapping genes": pl.List(pl.Utf8),
            "adjusted p-value": pl.Float64,
            "old p-value": pl.Float64,
            "old adjusted p-value": pl.Float64,
        },
    )

    return cast(list[EnrichrEntity], enriched_df.to_dicts())


def call_enrichr(
    genes: Sequence[str], gene_library: str = DEFAULT_GENE_LIBRARY
) -> list[EnrichrEntity]:
    """
    Call Enrichr (https://maayanlab.cloud/Enrichr/help#api)
    - asks enrichr to generate a list based on the genes provided
    - retrieves the generated list

    Args:
        genes (list[str]): list of genes
        gene_library (str): gene library to use; defaults to REACTOME_2022 (https://maayanlab.cloud/Enrichr/#libraries)
    """
    list_id = _generate_list(genes)
    entities = _retrieve_list(list_id, gene_library)
    return entities


def main():
    genes = [
        "BDNF",
        "MECP2",
        "PRRG1",
        "PTPN14",
        "PTPN21",
        "PTRF",
        "RAB23",
        "RAI14",
        "RASSF8",
        "RBFOX2",
        "RCN1",
        "RND3",
        "AFAP1",
        "AHNAK2",
        "AMOTL2",
        "ANTXR1",
        "ARHGAP23",
        "ARHGAP29",
        "ARSJ",
        "ASPH",
        "AXL",
        "CACNA2D1",
        "CALD1",
        "CALU",
        "CAP2",
        "CAV1",
        "CDR2L",
        "CHST3",
        "CLIC4",
        "CLIP2",
        "CLSTN1",
        "COL12A1",
        "COL4A1",
        "COL4A2",
        "COL5A2",
        "CRIM1",
        "CTGF",
        "CYR61",
        "DCBLD2",
        "DDAH1",
        "DKK1",
        "EDIL3",
        "EGFR",
        "EHD2",
        "ENAH",
        "ETV1",
        "FAT1",
        "FAT4",
        "FBN1",
        "FERMT2",
        "FGF2",
        "FHOD3",
        "FJX1",
        "FKBP10",
        "FLNC",
        "FOXL1",
        "FRMD6",
        "FSTL1",
        "GLI2",
        "GLRB",
        "GNAI1",
        "GNG12",
        "GPC1",
        "GPR161",
        "GPR176",
        "GPX8",
        "IFNWP19",
    ]
    entities = call_enrichr(genes, "KEGG_2015")
    print(entities)


if __name__ == "__main__":
    main()
