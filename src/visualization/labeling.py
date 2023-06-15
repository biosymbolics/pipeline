import numpy as np
from sklearn.decomposition import NMF
from gensim import corpora
from gensim.models.nmf import Nmf
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
import polars as pl
import logging
import streamlit as st
import matplotlib.pyplot as plt


MAX_FEATURES = 10000


def get_labels(df, n_features=MAX_FEATURES):
    tfidf_vectorizer = TfidfVectorizer(
        max_df=0.95,
        min_df=2,
        max_features=n_features,
        ngram_range=(1, 2),
        stop_words="english",
    )
    all_titles: list[str] = df.select(pl.col("title")).to_series().to_list()
    tfidf = tfidf_vectorizer.fit_transform(all_titles)
    coh_list = []
    # logging.info(all_titles)
    text = all_titles
    dictionary = corpora.Dictionary()
    corpus = dictionary.doc2bow(all_titles)
    n_topics = 3
    n_top_words = 15
    for n_topics in range(3, 5 + 1):
        print(n_topics, dictionary, corpus)
        # Train the model on the corpus
        nmf = Nmf(corpus, num_topics=n_topics, id2word=dictionary, random_state=42)

    nmf = NMF(n_components=n_topics, random_state=42, l1_ratio=0.5).fit(tfidf)
    nmf_embedding = nmf.transform(tfidf)
    feature_names = tfidf_vectorizer.get_feature_names_out()
    st.subheader("Topics found via NMF:")
    for topic_idx, topic in enumerate(nmf.components_):
        st.write("\nTopic {}:".format(topic_idx))
        st.write(
            " ".join(
                [
                    "[{}]".format(feature_names[i])
                    for i in topic.argsort()[: -n_top_words - 1 : -1]
                ]
            )
        )
    print()
