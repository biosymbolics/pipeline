"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
from typing import Any, Optional, cast
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


def get_default_tokenizer(nlp: Language):
    return Tokenizer(nlp.vocab)


class NerTagger:
    def __init__(
        self,
        model: str = "en_core_sci_scibert",  # alt models: en_core_sci_scibert, en_ner_bionlp13cg_md, en_ner_bc5cdr_md
        rule_sets: Optional[list[SpacyPatterns]] = None,
        get_tokenizer: Optional[GetTokenizer] = None,
    ):
        self.model = model
        self.rule_sets = (
            [
                INDICATION_SPACY_PATTERNS,
                INTERVENTION_SPACY_PATTERNS,
            ]
            if rule_sets is None
            else rule_sets
        )

        self.get_tokenizer = (
            get_default_tokenizer if get_tokenizer is None else get_tokenizer
        )

    def __create_tagger(self):
        nlp: Language = spacy.load(self.model)
        nlp.tokenizer = self.get_tokenizer(nlp)

        nlp.add_pipe("merge_entities", after="ner")
        ruler = nlp.add_pipe(
            "entity_ruler",
            config={"validate": True, "overwrite_ents": True},
            after="merge_entities",
        )
        for set in self.rule_sets:
            ruler.add_patterns(set)  # type: ignore

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

        self.nlp: Language = nlp

    def init(self):
        self.__create_tagger()

    def __call__(self, *args: Any, **kwds: Any) -> Any:
        if not self.nlp:
            self.__create_tagger()
        self.nlp(*args, **kwds)


def extract_named_entities(
    content: list[str],
    get_tokenizer: Optional[GetTokenizer] = None,
    rule_sets: Optional[list[SpacyPatterns]] = None,
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
    tagger = NerTagger(rule_sets=rule_sets, get_tokenizer=get_tokenizer)

    docs = [tagger(batch) for batch in content]
    entities = flatten([doc.ents for doc in docs])
    enriched = enrich_with_canonical(entities, nlp=tagger.nlp)
    entity_names = clean_entities(list(enriched.keys()), common_nlp)

    logging.info("Entity names: %s", entity_names)
    # debug_pipeline(docs, nlp)

    return entity_names
