"""
Named-entity recognition using spacy
"""
import re
from typing import Literal
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.tokens import Doc, Span
from spacy.language import Language
from spacy.tokenizer import Tokenizer
from spacy import displacy
from spacy.util import compile_prefix_regex, compile_suffix_regex
from pydash import flatten
import logging
from spacy_html_tokenizer import create_html_tokenizer

from common.ner.types import is_sci_spacy_linker
from constants.umls import UMLS_PHARMACOLOGIC_INTERVENTION_TYPES

from . import patterns
from .types import KbLinker

ENTITY_TYPES = ["PRODUCT"]
UNWRAP_TAGS = ["em", "strong", "b", "i", "span", "a", "code", "kbd", "li"]


def __add_tokenization_re(
    nlp: Language, re_type: Literal["infixes", "prefixes", "suffixes"], new_re: str
) -> list[str]:
    """
    Add regex to the tokenizer suffixes
    """
    if hasattr(nlp.Defaults, re_type):
        tokenizer_re_strs = list(getattr(nlp.Defaults, re_type))
        tokenizer_re_strs.append(new_re)
        return tokenizer_re_strs

    logging.warning(f"Could not find {re_type} in nlp.Defaults")
    return [new_re]


def custom_tokenizer(nlp: Language) -> Tokenizer:
    suffix_re = __add_tokenization_re(nlp, "suffixes", "®")
    prefix_re = __add_tokenization_re(nlp, "prefixes", "•")
    tokenizer = nlp.tokenizer
    tokenizer.suffix_search = compile_suffix_regex(suffix_re).search
    tokenizer.prefix_search = compile_prefix_regex(prefix_re).search
    return tokenizer


def __get_kb_linker(nlp: Language) -> KbLinker:
    """
    Get the KB linker from the nlp pipeline
    """
    linker = nlp.get_pipe("scispacy_linker")

    if not is_sci_spacy_linker(linker):
        raise Exception("Invalid linker")

    return linker.kb


def __has_intersection(list_a, list_b):
    return any(elem in list_a for elem in list_b)


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
            if __has_intersection(
                ent.types, UMLS_PHARMACOLOGIC_INTERVENTION_TYPES.keys()
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
    # train_ner(content)
    nlp: Language = spacy.load("en_core_sci_scibert")
    nlp.tokenizer = custom_tokenizer(nlp)
    nlp.tokenizer = create_html_tokenizer(unwrap_tags=UNWRAP_TAGS)(
        nlp
    )  # improves parsing of HTML

    nlp.add_pipe("merge_entities", before="ner")
    ruler = nlp.add_pipe("entity_ruler", config={"validate": True}, before="ner")
    print(patterns.BIOLOGICAL_PATTERNS)
    entity_patterns = [
        *[{"label": "PRODUCT", "pattern": moa_re} for moa_re in patterns.MOA_PATTERNS],
        *[
            {"label": "PRODUCT", "pattern": id_re}
            for id_re in patterns.INVESTIGATIONAL_ID_PATTERNS
        ],
        *[
            {"label": "PRODUCT", "pattern": bio_re}
            for bio_re in patterns.BIOLOGICAL_PATTERNS
        ],
    ]

    ruler.add_patterns(entity_patterns)  # type: ignore

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
    entities = flatten([doc.ents for doc in docs])
    linker = __get_kb_linker(nlp)
    canonical_entities = __get_canonical_entities(entities, linker)

    entity_strings = [
        entity.text for entity in entities if entity.label_ in ENTITY_TYPES
    ]
    # print(canonical_entities)
    displacy.serve(docs, style="ent", options={"fine_grained": True}, port=3333)  # type: ignore
    return entity_strings
