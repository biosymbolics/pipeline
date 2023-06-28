"""
Functions for copying around subsets of the patents database
"""
from clients.low_level.big_query import query_to_bg_table, BQ_DATASET_ID

from ._constants import COMMON_ENTITY_NAMES

BIOMEDICAL_IPC_CODES = ["A61", "C07", "C12", "G01N"]
IPC_RE = r"^({})".format("|".join(BIOMEDICAL_IPC_CODES))


def __copy_gpr_publications():
    """
    Copy publications from GPR to a local table
    """
    query = f"""
        SELECT * FROM `patents-public-data.google_patents_research.publications`
        WHERE EXISTS
        (SELECT 1 FROM UNNEST(cpc) AS cpc_code WHERE REGEXP_CONTAINS(cpc_code.code, "{IPC_RE}"))
    """
    query_to_bg_table(query, "gpr_publications")


def __copy_gpr_annotations():
    """
    Copy annotations from GPR to a local table

    To remove annotations after load:
    ``` sql
    UPDATE `patents.entities`
    SET annotations = ARRAY(
        SELECT AS STRUCT *
        FROM UNNEST(annotations) as annotation
        WHERE annotation.domain NOT IN ('chemClass', 'chemGroup', 'anatomy')
    )
    WHERE EXISTS(
        SELECT 1
        FROM UNNEST(annotations) AS annotation
        WHERE annotation.domain IN ('chemClass', 'chemGroup', 'anatomy')
    )
    ```

    or from gpr_annotations:
    ``` sql
    DELETE FROM `fair-abbey-386416.patents.gpr_annotations` where domain in
    ('chemClass', 'chemGroup', 'anatomy') OR preferred_name in ("seasonal", "behavioural", "mental health")
    ```
    """
    SUPPRESSED_DOMAINS = (
        "anatomy",
        "chemCompound",  # 961,573,847
        "chemClass",
        "chemGroup",
        "inorgmat",
        "methods",  # lots of useless stuff
        "nutrients",
        "nutrition",  # 109,587,438
        "polymers",
        "toxicity",  # 6,902,999
        "natprod",  # 23,053,704
        "species",  # 179,305,306
        "substances",  # 1,712,732,614
    )
    query = f"""
        SELECT annotations.* FROM `patents-public-data.google_patents_research.annotations` as annotations
        JOIN `{BQ_DATASET_ID}.publications` AS local_publications
        ON local_publications.publication_number = annotations.publication_number
        WHERE annotations.confidence > 0.69
        AND LOWER(preferred_name) not in {COMMON_ENTITY_NAMES}
        AND domain not in {SUPPRESSED_DOMAINS}
    """
    query_to_bg_table(query, "gpr_annotations")


def __copy_publications():
    """
    Copy publications from patents-public-data to a local table
    """
    query = f"""
        SELECT publications.* FROM `patents-public-data.patents.publications` as publications
        JOIN `{BQ_DATASET_ID}.gpr_publications` AS local_gpr
        ON local_gpr.publication_number = publications.publication_number
        WHERE application_kind = 'W'
    """
    query_to_bg_table(query, "publications")


def copy_patent_tables():
    """
    Copy tables from patents-public-data to a local dataset

    Order matters. Non-idempotent.
    """
    # copy gpr_publications table
    __copy_gpr_publications()

    # copy publications table
    __copy_publications()

    # copy gpr_annotations table
    __copy_gpr_annotations()
