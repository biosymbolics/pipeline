import numpy as np
from sklearn.decomposition import NMF
from gensim import corpora
from gensim.models.nmf import Nmf
from sklearn.feature_extraction.text import TfidfVectorizer
import polars as pl
import streamlit as st
import logging

from common.utils.dataframe import find_string_array_columns


MAX_FEATURES = 10000
N_TOP_WORDS = 15
N_TOPICS = 5  # TODO: coherence model - https://www.kaggle.com/code/yohanb/nmf-visualized-using-umap-and-bokeh/notebook


def get_labels_for_patents(df: pl.DataFrame):
    """
    Get the labels for the patents

    Args:
        df (pl.DataFrame): DataFrame
    """
    all_titles: list[str] = df.select(pl.col("title")).to_series().to_list()
    all_tags: list[str] = (
        df.select(pl.concat_list(find_string_array_columns(df))).to_series().to_list()
    )
    content = [*all_titles, *all_tags]
    get_labels(content)


def get_labels(content: list[str], n_features=MAX_FEATURES):
    """
    Get labels based on TDIDF and NMF

    Args:
        content (list[str]): List of strings
        n_features (int, optional): Number of features. Defaults to MAX_FEATURES.
    """

    logging.info("Extracting tf-idf features for NMF...")
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=n_features,
        ngram_range=(1, 2),
        stop_words="english",
    )
    tfidf = tfidf_vectorizer.fit_transform(content)
    dictionary = corpora.Dictionary()  # TODO
    corpus = dictionary.doc2bow(content)

    logging.info("Fitting the NMF model with tf-idf features")
    nmf = Nmf(corpus, num_topics=N_TOPICS, id2word=dictionary, random_state=42)
    nmf = NMF(n_components=N_TOPICS, random_state=42, l1_ratio=0.5).fit(tfidf)

    nmf_embedding = nmf.transform(tfidf)
    feature_names = tfidf_vectorizer.get_feature_names_out()

    st.subheader("Topics:")
    for topic_idx, topic in enumerate(nmf.components_):
        st.write("\nTopic {}:".format(topic_idx))
        st.write(
            " ".join(
                [
                    "[{}]".format(feature_names[i])
                    for i in topic.argsort()[: -N_TOP_WORDS - 1 : -1]
                ]
            )
        )
