"""
Named-entity recognition using spacy

No hardware acceleration: see https://github.com/explosion/spaCy/issues/10783#issuecomment-1132523032
"""
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.tokens import Span
from spacy.language import Language
from pydash import flatten

from common.ner.types import is_sci_spacy_linker
from constants.umls import UMLS_PHARMACOLOGIC_INTERVENTION_TYPES
from common.ner.utils import get_sec_tokenizer
from common.utils.list import has_intersection

from .debugging import debug_pipeline
from .patterns import INDICATION_SPACY_PATTERNS, INTERVENTION_SPACY_PATTERNS
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


def __enrich_with_canonical(
    entities: list[Span], kb_linker: KbLinker
) -> dict[str, list[str]]:
    """
    Links canonical entities if possible

    Args:
        entities (list[Span]): list of entities
        kb_linker (KbLinker): KB linker

    Currently only for PRODUCT entities
    """
    product_entities = [entity for entity in entities if entity.label_ in ENTITY_TYPES]

    canonical_entity_map = {}
    for entity in product_entities:
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
        canonical_entity_map[entity.text] = canonical_entities

    return canonical_entity_map


SUPPRESSIONS = ["phase", "trial"]


def __filter(entity: str) -> bool:
    is_suppressed = any([sup in entity.lower() for sup in SUPPRESSIONS])
    return not is_suppressed


def __clean_entity_names(entity_map: dict[str, list[str]]) -> list[str]:
    """
    Clean entity name list

    Args:
        entity_map (dict[str, list[str]]): entity map
    """
    # entity_names = [
    #     entity_map[entity][0] if entity_map[entity] else entity
    #     for entity in entity_map.keys()
    # ]
    entity_names = list(entity_map.keys())
    filtered = [entity for entity in entity_names if __filter(entity)]
    return filtered


def extract_named_entities(content: list[str]) -> list[str]:
    """
    Extract named entities from a list of content

    Args:
        content (list[str]): list of content on which to do NER
    """

    # alt models: en_core_sci_scibert, en_ner_bionlp13cg_md, en_ner_bc5cdr_md
    nlp: Language = spacy.load("en_core_sci_scibert")
    nlp.tokenizer = get_sec_tokenizer(nlp)

    nlp.add_pipe("merge_entities", after="ner")

    ruler = nlp.add_pipe(
        "entity_ruler",
        config={"validate": True, "overwrite_ents": True},
        after="merge_entities",
    )

    # order intentional
    ruler.add_patterns(INDICATION_SPACY_PATTERNS)  # type: ignore
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

    docs = [nlp(batch) for batch in content]

    entities = flatten([doc.ents for doc in docs])
    linker = __get_kb_linker(nlp)
    enriched = __enrich_with_canonical(entities, linker)
    entity_names = __clean_entity_names(enriched)

    print(entity_names)
    # debug_pipeline(docs, nlp)

    return entity_names
