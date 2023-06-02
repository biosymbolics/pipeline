"""
Contains the functions to train the NER model.
"""
import spacy
from skweak import base, doclevel, heuristics, generative, aggregation
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.matcher import Matcher
from spacy.language import Language
from spacy.tokens import Doc
import logging

from constants.patterns import moa as MoaPatterns, INDICATION_WORDS

from . import patterns

ADDITIONAL_INTERVENTION_WORDS = ["therapy", "treatment", "drug", "product"]


def __get_intervention_matcher(nlp):
    matcher = Matcher(nlp.vocab)
    matcher.add("CHEMICAL", patterns.ALL_PATTERNS)
    return matcher


def train_chem_and_disease_ner(nlp: Language, content: list[str]):
    docs = list(nlp.pipe(content))

    def annotate(doc: Doc):
        matcher = __get_intervention_matcher(nlp)
        matches = matcher(doc)
        for match_id, start, end in matches:
            yield start, end, "CHEMICAL"

    logging.info("Creating FunctionAnnotator")
    lf1 = heuristics.FunctionAnnotator("product", annotate)

    disease_words = [*INDICATION_WORDS, *[word.lower() for word in INDICATION_WORDS]]
    moa_words = [
        *MoaPatterns.MOA_SUFFIXES,
        *[term.title() for term in MoaPatterns.MOA_SUFFIXES],
        *ADDITIONAL_INTERVENTION_WORDS,
        *[word.lower() for word in ADDITIONAL_INTERVENTION_WORDS],
    ]
    moa_word_dict = {word: "CHEMICAL" for word in moa_words}
    disease_word_dict = {word: "DISEASE" for word in disease_words}
    lf2 = heuristics.VicinityAnnotator(
        "person_detector",
        {**moa_word_dict, **disease_word_dict},
        "nnp_detector",
        max_window=4,
    )

    maj_voter = aggregation.MajorityVoter(
        "doclevel_voter", ["CHEMICAL", "DISEASE"], initial_weights={"doc_majority": 0.0}
    )  # We do not want to include doc_majority itself in the vote

    combined = base.CombinedAnnotator()
    combined.add_annotator(lf1)
    combined.add_annotator(lf2)
    docs = list(combined.pipe(docs))
    docs = list(maj_voter.pipe(docs))

    print("PRODUCT", docs[0].spans["product"])
    print("VOTER", docs[0].spans["doclevel_voter"])

    logging.info("Aggregation started")

    # create and fit the HMM aggregation model
    hmm = generative.HMM("hmm", ["CHEMICAL", "DISEASE"])
    hmm.fit(docs)
    logging.info("Aggregation complete")

    hmm.fit_and_aggregate(docs)
    logging.info("Fit and aggregate complete")

    # save/load/return
