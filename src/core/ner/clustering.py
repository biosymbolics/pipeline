"""
Clustering utils
"""

from functools import reduce
import logging
from typing import Mapping, Sequence
import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer


from .constants import MAX_CLUSTER_SIZE, MAX_DISTINCT_TERMS, SOLO_CLUSTER_THRESHOLD


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def fit_tfidf_vectorizer(
    terms: Sequence[str],
    common_words: set[str] = set([]),
    common_word_weight: float = 0.1,
    ngram_range: tuple[int, int] = (1, 1),
) -> TfidfVectorizer:
    """
    Fits a TF-IDF vectorizer with a custom IDF for common words

    Args:
        terms (Sequence[str]): list of terms to fit
        common_words (set[str]): set of common words to lower the IDF for
        common_word_weight (float): weight to lower the IDF for common words (default: 0.1)
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=ngram_range,
        strip_accents="unicode",
    )
    vectorizer.fit(terms)

    # set IDFs lower for known-common words
    vectorizer.idf_ = list(
        {
            k: ((v * common_word_weight) if k in common_words else v)
            for k, v in zip(vectorizer.get_feature_names_out(), vectorizer.idf_)
        }.values()
    )

    logger.info("Fitted TF-IDF vectorizer")

    return vectorizer


def _create_cluster_term_map(
    terms: Sequence[str],
    counts: Sequence[int],
    cluster_ids: Sequence[int],
    max_cluster_size: int = MAX_CLUSTER_SIZE,
    max_distinct_terms: int = MAX_DISTINCT_TERMS,
) -> tuple[dict[str, str], list[dict[str, int]]]:
    """
    Creates a map between synonym/secondary terms and the primary
    For use with cluster_terms

    Args:
        terms (Sequence[str]): list of terms to cluster
        counts (Sequence[int]): list of counts for the terms (used for canonical name selection)
        cluster_ids (Sequence[int]): list of cluster ids
        max_cluster_size (int): maximum cluster size (default: MAX_CLUSTER_SIZE)
        max_distinct_terms (int): maximum distinct terms in a cluster (default: MAX_DISTINCT_TERMS)
    """
    df = (
        pl.DataFrame({"cluster_id": cluster_ids, "name": terms, "count": counts})
        .group_by("cluster_id")
        .agg(
            [
                pl.col("count").sum().alias("cluster_count"),
                pl.concat_list(pl.col("name").str.split(" "))
                .flatten()
                .unique()
                .count()
                .alias("distinct_terms"),
                pl.col("name"),
                pl.col("count"),
            ]
        )
        .with_columns(
            is_other=pl.when(
                pl.col("cluster_count").gt(max_cluster_size)
                | pl.col("distinct_terms").gt(max_distinct_terms)
                | pl.col("cluster_id").eq(-1)
            )
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
        )
        .explode(["name", "count"])
        .group_by(["cluster_id", "name"])
        .agg([pl.col("count").sum().alias("count"), pl.col("is_other").max()])
        .sort("count", descending=True)  # sort so that first name is the most common
    )
    term_clusters = (
        df.filter(pl.col("is_other") == False)
        .group_by("cluster_id", maintain_order=True)
        .agg([pl.col("name"), pl.col("count").sum().alias("count")])
        .drop("cluster_id")
        .to_series()
        .to_list()
    )

    # remaining terms, grouped by cluster_id.
    others = [
        dict(o.select("name", "count").iter_rows())
        for o in df.filter(pl.col("is_other") == True).partition_by("cluster_id")
    ]

    return {
        m: members_terms[0]  # synonym to most-frequent-term
        for members_terms in term_clusters
        for m in members_terms
    }, others


def cluster_terms_with_hdbscan(
    terms: Sequence[str],
    counts: Sequence[int],
    common_words: set[str] = set([]),
    max_cluster_size: int = MAX_CLUSTER_SIZE,
) -> dict[str, str]:
    """
    Clusters terms using TF-IDF and HDBSCAN.
    Returns a synonym record mapping between the canonical term and the synonym (one per pair).

    Is a memory pig but has good accuracy. Used for the remaining terms after DBSCAN.

    Args:
        terms (Sequence[str]): list of terms to cluster
        counts (Sequence[int]): list of counts for the terms
                (used only for canonical name selection, since HDBSCAN impl doesn't support sample weights)
        common_words (set[str]): set of common words to lower the IDF for (default: set([]))
        max_cluster_size (int): maximum cluster size (default: MAX_CLUSTER_SIZE)

    Returns:
        dict[str, str]: map between synonym/secondary terms and the primary
    """
    # lazy loading
    from sklearn.cluster._hdbscan.hdbscan import HDBSCAN

    vectorizer = fit_tfidf_vectorizer(terms, common_words=common_words)
    vectorized = vectorizer.transform(terms)

    logger.info("Starting HDBSCAN clustering (shape %s)", vectorized.shape)
    cluster_ids = (
        HDBSCAN(
            # merge clusters that are close
            # https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#selecting-cluster-selection-epsilon
            cluster_selection_epsilon=0.4,
            # https://hdbscan.readthedocs.io/en/latest/parameter_selection.html#leaf-clustering
            cluster_selection_method="leaf",
            min_cluster_size=3,
            max_cluster_size=max_cluster_size,
        )
        .fit_predict(vectorized)
        .tolist()
    )

    result_map, _ = _create_cluster_term_map(terms, counts, cluster_ids)
    return result_map


def cluster_terms(
    term_count_map: Mapping[str, int],
    common_words: set[str] = set([]),
    recluster_remaining: bool = True,
    solo_cluster_threshold: int = SOLO_CLUSTER_THRESHOLD,
) -> dict[str, str]:
    """
    Clusters terms using TF-IDF and DBSCAN. Sends remaining through a HDBSCAN recluster.
    Returns a synonym record mapping between the canonical term and the synonym (one per pair).

    Args:
        term_count_map (Mapping[str, int]): map between term and count
        common_words (set[str]): set of common words to lower the IDF for (default: set([]))

    Returns:
        dict[str, str]: map between synonym/secondary terms and the primary
    """
    # lazy loading
    from sklearn.cluster import DBSCAN

    terms = list(term_count_map.keys())
    counts = list(term_count_map.values())

    vectorizer = fit_tfidf_vectorizer(terms, common_words=common_words)
    vectorized = vectorizer.transform(terms)

    logger.info("Starting clustering (shape %s)", vectorized.shape)
    # HDBSCAN is *much better* but a memory pig
    cluster_ids = (
        DBSCAN(eps=0.7, min_samples=2).fit_predict(vectorized, counts).tolist()
    )

    result_map, others = _create_cluster_term_map(terms, counts, cluster_ids)
    logger.info("Found %s clusters", len(result_map))

    # if true, go through another pass on the remaining terms
    if recluster_remaining:
        logger.info("Reclustering remaining terms (%s)", sum(len(o) for o in others))
        # TODO: use HDSCAN if cluster is sufficiently small (~40k expanded)
        remaining_cluster_map: dict[str, str] = reduce(
            lambda x, other: {
                **x,
                **cluster_terms(
                    other,
                    common_words,
                    recluster_remaining=False,
                    solo_cluster_threshold=solo_cluster_threshold,
                ),
            },
            others,
            {},
        )

        return {**result_map, **remaining_cluster_map}

    # turns remaining entries into their own category, if sufficiently large
    # otherwise, marks as "other"
    remaining_cluster_map = {
        k: (k if v >= solo_cluster_threshold else "other")
        for o in others
        for k, v in o.items()
    }
    return {**result_map, **remaining_cluster_map}
