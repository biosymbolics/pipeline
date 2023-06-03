"""
Named-entity recognition using spacy
"""
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.tokens import Span
from spacy.language import Language
from spacy import displacy
from pydash import flatten
import logging

from common.ner.types import is_sci_spacy_linker
from constants.umls import UMLS_PHARMACOLOGIC_INTERVENTION_TYPES
from common.ner.utils import get_sec_tokenizer
from common.utils.list import has_intersection

from .debugging import print_tokens, serve_ner_viewer
from .patterns import INTERVENTION_SPACY_PATTERNS
from .types import KbLinker

ENTITY_TYPES = ["PRODUCT"]


def __get_kb_linker(nlp: Language) -> KbLinker:
    """
    Get the KB linker from the nlp pipeline
    """
    linker = nlp.get_pipe("scispacy_linker")

    if not is_sci_spacy_linker(linker):
        raise Exception("Invalid linker")

    return linker.kb


def __get_canonical_entities(entities: list[Span], kb_linker: KbLinker) -> dict:
    """
    Get canonical entities from the entities
    """
    canonical_entity_map = {}
    for entity in entities:
        kb_entities = [
            kb_linker.cui_to_entity[kb_ent[0]] for kb_ent in entity._.kb_ents
        ]
        canonical_entities = [
            ent.canonical_name
            for ent in kb_entities
            if has_intersection(
                ent.types, list(UMLS_PHARMACOLOGIC_INTERVENTION_TYPES.keys())
            )
        ]
        if len(canonical_entities) > 0:
            canonical_entity_map[entity.text] = canonical_entities

    return canonical_entity_map


def extract_named_entities(content: list[str]) -> list[str]:
    """
    Extract named entities from a list of content

    Args:
        content (list[str]): list of content on which to do NER

    TODO:
    - POS tagging on SciSpacy
    """

    # alt models: en_core_sci_scibert, en_ner_bionlp13cg_md, en_ner_bc5cdr_md
    nlp: Language = spacy.load("en_ner_bc5cdr_md")
    nlp.tokenizer = get_sec_tokenizer(nlp)

    # choosing not to do this (otherwise patterns will require .*)
    # nlp.add_pipe("merge_entities", after="ner")

    ruler = nlp.add_pipe(
        "entity_ruler",
        config={"validate": True, "overwrite_ents": True},
    )
    ruler.add_patterns(INTERVENTION_SPACY_PATTERNS)  # type: ignore

    nlp.add_pipe(
        "scispacy_linker",
        config={
            "resolve_abbreviations": True,
            "linker_name": "umls",
            "threshold": 0.7,
            "filter_for_definitions": False,
            "no_definition_threshold": 0.7,
        },
    )
    analysis = nlp.analyze_pipes(pretty=True)
    logging.debug("About the pipeline: %s", analysis)
    docs = [nlp(batch) for batch in content]

    print_tokens(docs)

    entities = flatten([doc.ents for doc in docs])
    linker = __get_kb_linker(nlp)
    canonical_entities = __get_canonical_entities(entities, linker)

    # entity_strings = [
    #     entity.text for entity in entities if entity.label_ in ENTITY_TYPES
    # ]
    # print(canonical_entities)
    serve_ner_viewer(docs)
    return []
