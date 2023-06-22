"""
Contains the functions to train the NER model.
"""
from functools import partial
from skweak import base, heuristics, generative, aggregation, utils
from spacy.matcher import Matcher
from spacy.language import Language
from spacy.tokens import Doc
import logging

from . import patterns

ENTITITES = {
    "INDICATION": patterns.INDICATION_PATTERNS,
    "INTERVENTION": patterns.ALL_INTERVENTION_PATTERNS,
}


def annotate(doc: Doc, nlp: Language):
    """
    Annotate a document with a label and patterns

    Args:
        doc (Doc): SpaCy document
        nlp (Language): SpaCy language model
    """
    matcher = Matcher(nlp.vocab)

    for label, patterns in ENTITITES.items():
        matcher.add(label, patterns)

    matches = matcher(doc, as_spans=True)

    for span in matches:
        yield span.start, span.end, span.label_


def weakly_train_ner(nlp: Language, content: list[str], filename: str):
    """
    Train the NER model using weak supervision

    Args:
        nlp (Language): SpaCy language model
        content (list[str]): List of strings to train on
        filename (str): where to save the output (labeled training set)

    TODO:
        - specificity patterns?
        - majority voter - least common denominator isn't useful.
    """
    docs = list(nlp.pipe(content))

    logging.info("Creating FunctionAnnotator")
    lf1 = heuristics.FunctionAnnotator(
        "entities",
        partial(
            annotate,
            nlp=nlp,
        ),
    )

    maj_voter = aggregation.MajorityVoter(
        "doclevel_voter",
        list(ENTITITES.keys()),
        initial_weights={"doc_majority": 0.0},
    )  # We do not want to include doc_majority itself in the vote

    combined = base.CombinedAnnotator()
    combined.add_annotator(lf1)
    docs = list(combined.pipe(docs))
    docs = list(maj_voter.pipe(docs))

    logging.info("Entities on doc 0: %s", docs[0].spans["entities"])
    logging.info("Entities on doc 0 (VOTER): %s", docs[0].spans["doclevel_voter"])

    logging.info("Aggregation started")
    hmm = generative.HMM("hmm", list(ENTITITES.keys()))
    docs = hmm.fit_and_aggregate(docs)

    logging.info("Fit and aggregate complete")

    for doc in docs:
        doc.ents = doc.spans["hmm"]

    utils.docbin_writer(docs, filename)
