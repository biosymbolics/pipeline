import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
import polars as pl

from system import initialize

initialize()

from clients.low_level.postgres import PsqlDatabaseClient
from scripts.patents.utils import clean_owners
from utils.list import dedup

COMPANY_DEDUP_FILE = "company_dedup.csv"


def dedup_companies():
    query = """
        select lower(company_name) as company_name from (
            select distinct sponsor as company_name from trials

            UNION ALL

            select assignee.name as company_name
            FROM applications a,
            unnest(a.assignees) as assignee
        ) s
        group by lower(company_name)
        having count(*) > 10
    """
    records = PsqlDatabaseClient().select(query)
    org_names = [record["company_name"] for record in records]
    org_names_norm = dedup(clean_owners(org_names))
    vectorizer = TfidfVectorizer(stop_words="english", strip_accents="unicode")
    X = vectorizer.fit_transform(org_names_norm)
    clustering = DBSCAN(eps=0.8, min_samples=2).fit(X)
    labels = clustering.labels_

    df = pl.DataFrame({"id": labels, "company_name": org_names_norm})
    grouped_cos = (
        df.filter(pl.col("id") > -1)
        .groupby("id")
        .agg(pl.col("company_name"))
        .drop("id")
        .to_series()
        .to_list()
    )

    synonyms = [
        {"term": members[0], "synonym": m}
        for members in grouped_cos
        for m in members[1:]
    ]
    records = PsqlDatabaseClient().insert_into_table(synonyms, "synonym_map")


def main():
    """
    Dedup
    """
    dedup_companies()


if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 -m scripts.patents.dedup_companies")
        sys.exit()

    main()
