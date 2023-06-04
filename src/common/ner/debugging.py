"""
Debugging functions for NER
"""
import multiprocessing
import os
from typing import Literal
from spacy.tokens import Doc
from spacy import displacy
from spacy.language import Language
import logging

ViewerStyle = Literal["ent", "dep", "span"]


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


def __serve_displacy(docs: list[Doc], style: ViewerStyle, port: int):
    """
    Serves the displacy viewer
    """
    displacy.serve(
        docs,
        style=style,
        options={
            "fine_grained": True,
            "add_lemma": True,
            "colors": {"DISEASE": "pink", "CHEMICAL": "green"},
        },
        port=port,
    )


def serve_ner_viewer(docs: list[Doc], styles: list[ViewerStyle] = ["ent", "dep"]):
    """
    Serves the NER viewer

    Args:
        docs (list[Doc]): list of spacy docs

    TODO: type "dep" does not work with en_ner_bc5cdr_md ("assignment destination is read-only")
    """
    init_port = 3332
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    processes = []
    for idx, style in enumerate(styles):
        my_docs = docs[:]
        process = multiprocessing.Process(
            target=__serve_displacy, args=(my_docs, style, init_port + idx)
        )
        processes.append(process)
        process.start()

    for process in processes:
        process.join()


def debug_pipeline(nlp: Language):
    """
    Logs debugging info about the nlp pipeline

    Args:
        nlp (Language): spacy nlp pipeline
    """
    analysis = nlp.analyze_pipes(pretty=True)
    logging.debug("About the pipeline: %s", analysis)
