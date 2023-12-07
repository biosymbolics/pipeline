from typing import Optional
import json
import datetime
import logging

import scipy
import numpy
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from scispacy.linking_utils import (
    KnowledgeBase,
    UmlsKnowledgeBase,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def create_tfidf_ann_index(
    out_path: str, kb: Optional[KnowledgeBase] = None
) -> tuple[list[str], TfidfVectorizer]:
    """
    Build tfidf vectorizer and ann index.

    Parameters
    ----------
    out_path: str, required.
        The path where the various model pieces will be saved.
    kb : KnowledgeBase, optional.
        The kb items to generate the index and vectors for.

    """
    tfidf_vectorizer_path = f"{out_path}/tfidf_vectorizer.joblib"
    ann_index_path = f"{out_path}/nmslib_index.bin"
    tfidf_vectors_path = f"{out_path}/tfidf_vectors_sparse.npz"
    umls_concept_aliases_path = f"{out_path}/concept_aliases.json"

    kb = kb or UmlsKnowledgeBase()

    logger.info(
        "No tfidf vectorizer on %s or ann index on %s",
        tfidf_vectorizer_path,
        ann_index_path,
    )
    concept_aliases = list(kb.alias_to_cuis.keys())

    # NOTE: here we are creating the tf-idf vectorizer with float32 type, but we can serialize the
    # resulting vectors using float16, meaning they take up half the memory on disk. Unfortunately
    # we can't use the float16 format to actually run the vectorizer, because of this bug in sparse
    # matrix representations in scipy: https://github.com/scipy/scipy/issues/7408
    logger.info("Fitting tfidf vectorizer on %s aliases", len(concept_aliases))
    tfidf_vectorizer = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2), min_df=10, dtype=numpy.float32
    )
    start_time = datetime.datetime.now()
    concept_alias_tfidfs = tfidf_vectorizer.fit_transform(concept_aliases)
    logger.info("Saving tfidf vectorizer to %s", tfidf_vectorizer_path)
    joblib.dump(tfidf_vectorizer, tfidf_vectorizer_path)

    total_time = datetime.datetime.now() - start_time
    logger.info(
        "Fitting and saving vectorizer took %s seconds", total_time.total_seconds()
    )

    logger.info(
        "Saving list of concept ids and tfidfs vectors to %s and %s",
        umls_concept_aliases_path,
        tfidf_vectors_path,
    )
    json.dump(concept_aliases, open(umls_concept_aliases_path, "w"))
    scipy.sparse.save_npz(
        tfidf_vectors_path, concept_alias_tfidfs.astype(numpy.float16)
    )

    return concept_aliases, tfidf_vectorizer
