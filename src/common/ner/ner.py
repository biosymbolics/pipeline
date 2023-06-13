"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.language import Language
from pydash import flatten
import logging

from common.ner.utils import get_sec_tokenizer

from .debugging import debug_pipeline
from .linking import enrich_with_canonical
from .patterns import INDICATION_SPACY_PATTERNS, INTERVENTION_SPACY_PATTERNS
from .cleaning import clean_entities

common_nlp = spacy.load("en_core_web_sm")

# loading here because this takes about 6 seconds per invocation
# alt models: en_core_sci_scibert, en_ner_bionlp13cg_md, en_ner_bc5cdr_md
sci_nlp: Language = spacy.load("en_core_sci_scibert")


def extract_named_entities(content: list[str]) -> list[str]:
    """
    Extract named entities from a list of content

    Args:
        content (list[str]): list of content on which to do NER
    """
    sci_nlp.tokenizer = get_sec_tokenizer(sci_nlp)

    sci_nlp.add_pipe("merge_entities", after="ner")

    ruler = sci_nlp.add_pipe(
        "entity_ruler",
        config={"validate": True, "overwrite_ents": True},
        after="merge_entities",
    )

    # order intentional
    ruler.add_patterns(INDICATION_SPACY_PATTERNS)  # type: ignore
    ruler.add_patterns(INTERVENTION_SPACY_PATTERNS)  # type: ignore

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

    docs = [sci_nlp(batch) for batch in content]

    entities = flatten([doc.ents for doc in docs])
    enriched = enrich_with_canonical(entities, nlp=sci_nlp)
    entity_names = clean_entities(enriched, common_nlp)

    logging.info("Entity names: %s", entity_names)
    # debug_pipeline(docs, nlp)

    return entity_names
