"""
Contains the functions to train the NER model.
"""
from functools import partial
from skweak import base, heuristics, generative, aggregation
from spacy.matcher import Matcher
from spacy.language import Language
from spacy.tokens import Doc
import logging

from . import patterns


def annotate(doc: Doc, label: str, patterns: list[list[dict]], nlp):
    matcher = Matcher(nlp.vocab)
    matcher.add(label, patterns)
    matches = matcher(doc)
    for match_id, start, end in matches:
        yield start, end, label


def train_ner(nlp: Language, content: list[str]):
    docs = list(nlp.pipe(content))

    annotate_indication = partial(
        annotate, label="INDICATION", patterns=patterns.INDICATION_PATTERNS, nlp=nlp
    )
    annotate_intervention = partial(
        annotate,
        label="INTERVENTION",
        patterns=patterns.ALL_INTERVENTION_PATTERNS,
        nlp=nlp,
    )

    logging.info("Creating FunctionAnnotator")
    lf1 = heuristics.FunctionAnnotator("intervention", annotate_intervention)
    lf2 = heuristics.FunctionAnnotator("indication", annotate_indication)

    maj_voter = aggregation.MajorityVoter(
        "doclevel_voter",
        ["INTERVENTION", "INDICATION"],
        initial_weights={"doc_majority": 0.0},
    )  # We do not want to include doc_majority itself in the vote

    combined = base.CombinedAnnotator()
    combined.add_annotator(lf1)
    combined.add_annotator(lf2)
    docs = list(combined.pipe(docs))
    docs = list(maj_voter.pipe(docs))

    print("INTERVENTION", docs[0].spans["intervention"])
    print("VOTER", docs[0].spans["doclevel_voter"])

    logging.info("Aggregation started")

    # create and fit the HMM aggregation model
    hmm = generative.HMM("hmm", ["INTERVENTION", "INDICATION"])
    hmm.fit(docs)
    logging.info("Aggregation complete")

    hmm.fit_and_aggregate(docs)
    logging.info("Fit and aggregate complete")

    # save/load/return
    hmm.save("hmm-inter-inda.pkl")
