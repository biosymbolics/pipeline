"""
Debugging functions for NER
"""
from typing import Literal
from spacy.tokens import Doc
from spacy import displacy
from spacy.language import Language
import logging


def print_tokens(docs: list[Doc], lemma_suffixes: list[str] = ["ucel", "mab", "nib"]):
    """
    Prints debug information about tokens

    Args:
        docs (list[Doc]): list of spacy docs
        lemma_suffixes (list[str]): list of suffixes to match
    """
    for token in [token for doc in docs for token in doc]:
        is_match = any([suffix in token.lemma_ for suffix in lemma_suffixes])
        if is_match:
            print(
                token.ent_type_ or "UNKNOWN",
                token.text,
                "tag:",
                token.tag_,
                "pos:",
                token.pos_,
                "dep:",
                token.dep_,
                "lemma:",
                token.lemma_,
                "morph:",
                token.morph,
                "prob:",
                token.prob,
                "head:",
                token.head.text,
                "span:",
                [child for child in token.children],
            )


ViewerStyle = Literal["ent", "dep"]


def serve_ner_viewer(docs: list[Doc], style: ViewerStyle = "ent"):
    """
    Serves the NER viewer

    Args:
        docs (list[Doc]): list of spacy docs
    """
    displacy.serve(docs, style=style, options={"fine_grained": True, "add_lemma": True}, port=3333)  # type: ignore


def debug_pipeline(nlp: Language):
    """
    Logs debugging info about the nlp pipeline

    Args:
        nlp (Language): spacy nlp pipeline
    """
    analysis = nlp.analyze_pipes(pretty=True)
    logging.debug("About the pipeline: %s", analysis)
