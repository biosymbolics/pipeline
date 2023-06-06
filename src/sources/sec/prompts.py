"""
LLM prompts for SEC doc parsing
"""

from typing import Mapping, TypedDict
from llama_index.prompts.prompts import KnowledgeGraphPrompt


class Schema(TypedDict):
    """
    Schema class stub
    """

    type: str
    properties: Mapping[str, dict]


JSON_PIPELINE_SCHEMA: Schema = {
    "type": "object",
    "properties": {
        "generic_name": {"type": "string"},
        "brand_name": {"type": "string"},
        # "indication": {"type": "string"},
        # "description": {"type": "string"},
    },
}


JSON_PIPELINE_PROMPT = (
    "What are the products in active research and development (R&D)? "
    "Return the results as an array of json objects containing the "
    "keys: " + ", ".join(JSON_PIPELINE_SCHEMA["properties"].keys()) + "."
)

BIOMEDICAL_TRIPLET_EXTRACT_TMPL = (
    "Sections of a pharmaceutical company SEC 10-K document is provided below as HTML. "
    "Extract information about all the company's products and related attributes in the form of "
    "knowledge triplets (subject, predicate, object). "
    "Attributes of interest include:\n"
    " - status: investigational, commercial, LOE, phase, submitted\n"
    " - indications\n"
    " - mechanisms of action\n"
    " - synonyms\n"
    "Avoid stop words and long predicates. "
    "Be consistent with subject, predictate, object naming and ordering. "
    "Return a maximum of {max_knowledge_triplets}.\n"
    "---------------------\n"
    "Example 1:\n"
    "Text: Alice is Bob's mother.\n"
    "Triplets: (Alice, is mother of, Bob)\n"
    "Example 2:\n"
    "Text: Phase III clinical trials are underway for cendakimab in eosinophilic esophagitis.\n"
    "Triplets:\n"
    "(cendakimab, has indication, Eosinophilic Esophagitis)\n"
    "(cendakimab, has status, Phase III)\n"
    "(cendakimab, has status, investigational)\n"
    "Example 3:\n"
    "Text: OPDUALAG (nivolumab + relatlimab) 1L Melanoma\n"
    "Triplets:\n"
    "(OPDUALAG, contains, nivolumab)\n"
    "(OPDUALAG, contains, relatlimab)\n"
    "(OPDUALAG, has indication, Melanoma)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)

BIOMEDICAL_TRIPLET_EXTRACT_PROMPT = KnowledgeGraphPrompt(
    BIOMEDICAL_TRIPLET_EXTRACT_TMPL
)

BIOMEDICAL_TRIPLET_EXTRACT_PROMPT = KnowledgeGraphPrompt(
    BIOMEDICAL_TRIPLET_EXTRACT_TMPL
)


def GET_BIOMEDICAL_NER_TEMPLATE(entity: str) -> str:
    return (
        f"Assuming '{entity}' is a pharmaceutical compound, mechanism of action or other intervention, do as follows: "
        "Return information about this intervention, such as its name, "
        "drug class, mechanism of action, target(s), indication(s), status, competition, novelty etc. "
        "- If investigational, include details about its phase of development and probability of success. "
        "- If approved, include details about its regulatory status, commercialization, revenue and prospects. "
        "- If discontinued, include the reasons for discontinuation. "
    )
