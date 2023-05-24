"""
Named-entity recognition using spacy
"""
import spacy
from scispacy.linking import EntityLinker  # required to use 'scispacy_linker' pipeline
from spacy.tokens import Span
from spacy.language import Language
from pydash import compact, flatten
import logging

from common.ner.types import is_sci_spacy_linker
from .types import KbLinker

ENTITY_TYPES = ["PRODUCT"]

SUMMARY_ATTRIBUTES = ["text", "label_", "kb_id_", "canonical_name", "types"]

type_map = {
    "T019": "Congenital Abnormality",
    "T022": "Body System",
    "T024": "Tissue",
    "T025": "Cell",
    "T028": "Gene or Genome",
    "T033": "Finding",
    "T037": "Injury or Poisoning",
    "T038": "Biologic Function",
    "T041": "Mental Process",
    "T044": "Molecular Function",
    "T045": "Genetic Function",
    "T046": "Pathologic Function",
    "T047": "Disease or Syndrome",
    "T048": "Mental or Behavioral Dysfunction",
    "T058": "Health Care Activity",
    "T059": "Laboratory Procedure",
    "T060": "Diagnostic Procedure",
    "T061": "Therapeutic or Preventive Procedure",
    "T067": "Phenomenon or Process",
    "T068": "Human-caused Phenomenon or Process",
    "T074": "Medical Device",
    "T086": "Nucleotide Sequence",
    "T087": "Amino Acid Sequence",
    "T109": "Organic Chemical",
    "T101": "Patient or Disabled Group",
    "T114": "Nucleic Acid, Nucleoside, or Nucleotide",
    "T116": "Amino Acid, Peptide, or Protein",
    "T121": "Pharmacologic Substance",
    "T123": "Biologically Active Substance",
    "T125": "Hormone",
    "T126": "Enzyme",
    "T127": "Vitamin",
    "T129": "Immunologic Factor",
    "T130": "Indicator, Reagent, or Diagnostic Aid",
    "T131": "Hazardous or Poisonous Substance",
    "T167": "Substance",
    "T184": "Sign or Symptom",
    "T190": "Anatomical Abnormality",
    "T191": "Neoplastic Process",
    "T195": "Antibiotic",
    "T197": "Inorganic Chemical",
    "T200": "Clinical Drug",
    "T203": "Drug Delivery Device",
}


def __summarize(entities: list[tuple]):
    for ent in entities:
        for attr in SUMMARY_ATTRIBUTES:
            if hasattr(ent, attr):
                print(f"Entity {attr} : {getattr(ent, attr)}")


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


def __get_kb_entities(entities: list[Span], kb_linker: KbLinker) -> list[str]:
    """
    Get mapped KB entities from the entities
    """
    kb_ents: list[Span] = flatten([entity._.kb_ents for entity in entities])
    linked_ents = [kb_linker.cui_to_entity[umls_ent[0]] for umls_ent in kb_ents]

    canonical_names = [
        ent.canonical_name if __has_intersection(ent.types, type_map.keys()) else None
        for ent in linked_ents
    ]
    return compact(canonical_names)


def extract_named_entities(content: list[str]) -> list[str]:
    """
    Extract named entities from a list of content

    Args:
        content (list[str]): list of content on which to do NER
    """
    nlp = spacy.load("en_core_sci_sm")  # en_core_web_trf
    nlp.add_pipe(
        "scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"}
    )
    entities = flatten([nlp(batch).ents for batch in content])

    linker = __get_kb_linker(nlp)
    kb_entities = __get_kb_entities(entities, linker)

    # __summarize(entities)
    # __summarize(kb_entities)
    print(kb_entities)

    entity_strings = [
        entity.text if entity.label_ in ENTITY_TYPES else "" for entity in entities
    ]
    return entity_strings
