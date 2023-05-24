import spacy
from skweak import heuristics, aggregation
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.matcher import Matcher
from spacy.language import Language
from spacy.tokens import Doc
import logging

from . import patterns


def __get_moa_matcher(nlp):
    matcher = Matcher(nlp.vocab)
    matcher.add("PRODUCT", [patterns.MOA_PATTERN])
    return matcher


def train_ner(content: list[str]):
    nlp: Language = spacy.load("en_core_sci_lg")
    nlp.add_pipe("merge_entities")

    def moa(doc: Doc):
        matcher = __get_moa_matcher(nlp)
        patterns = []
        matches = matcher(doc)
        for match_id, start, end in matches:
            patterns.append((start, end, "PRODUCT"))
        print(patterns)
        return patterns

    lf1 = heuristics.FunctionAnnotator("moa", moa)

    doc = nlp("\n".join(content))
    doc = lf1(doc)

    hmm = aggregation.HMM("hmm", ["PRODUCT"])
    logging.info("Aggregation complete")

    hmm.fit_and_aggregate([doc])
    logging.info("Fit and aggregate complete")
