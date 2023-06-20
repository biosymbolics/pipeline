"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
from typing import cast
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.language import Language
from spacy.pipeline.entityruler import EntityRuler
from spacy.tokenizer import Tokenizer
from pydash import flatten
import logging

from .cleaning import clean_entities
from .debugging import debug_pipeline
from .linking import enrich_with_canonical
from .patterns import INDICATION_SPACY_PATTERNS, INTERVENTION_SPACY_PATTERNS
from .types import GetTokenizer, SpacyPatterns

common_nlp = spacy.load("en_core_web_sm")

# loading here because this takes about 6 seconds per invocation
# alt models: en_core_sci_scibert, en_ner_bionlp13cg_md, en_ner_bc5cdr_md
sci_nlp: Language = spacy.load("en_core_sci_scibert")
sci_nlp.add_pipe("merge_entities", after="ner")
ruler: EntityRuler = cast(
    EntityRuler,
    sci_nlp.add_pipe(
        "entity_ruler",
        config={"validate": True, "overwrite_ents": True},
        after="merge_entities",
    ),
)
sci_nlp.add_pipe(
    "scispacy_linker",
    config={
        "resolve_abbreviations": True,
        "linker_name": "umls",
        "threshold": 0.7,
        "filter_for_definitions": False,
        "no_definition_threshold": 0.7,
    },
)


def get_default_tokenzier(nlp: Language):
    return Tokenizer(nlp.vocab)


def extract_named_entities(
    content: list[str],
    get_tokenizer: GetTokenizer = get_default_tokenzier,
    rule_sets: list[SpacyPatterns] = [
        INDICATION_SPACY_PATTERNS,
        INTERVENTION_SPACY_PATTERNS,
    ],
) -> list[str]:
    """
    Extract named entities from a list of content
    - basic SpaCy pipeline
    - applies rule_sets
    - applies scispacy_linker (canonical mapping to UMLS)

    Args:
        content (list[str]): list of content on which to do NER
        rule_sets (list[SpacyPatterns]): list of rule sets to apply
    """
    sci_nlp.tokenizer = get_tokenizer(sci_nlp)

    for set in rule_sets:
        ruler.add_patterns(set)  # type: ignore

    docs = [sci_nlp(batch) for batch in content]

    entities = flatten([doc.ents for doc in docs])
    enriched = enrich_with_canonical(entities, nlp=sci_nlp)
    entity_names = clean_entities(list(enriched.keys()), common_nlp)

    logging.info("Entity names: %s", entity_names)
    # debug_pipeline(docs, nlp)

    return entity_names
