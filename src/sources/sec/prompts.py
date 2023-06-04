"""
LLM prompts for SEC doc parsing
TODO:
- turn the schema/validation/prompt thing into reusable method
- train LLM on validation if it tends to be a problem (or fix)
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
    "From this, extract ALL of the company's products and their attributes as "
    "triplets (subject, predicate, object). "
    "Related attributes (objects) of interest include:\n"
    " - status: investigational, commercial, LOE, phase, submitted\n"
    " - indications\n"
    " - mechanisms of action\n"
    " - synonyms\n"
    "Avoid stop words and long predicates. "
    "Return a maximum of {max_knowledge_triplets}.\n"
    "---------------------\n"
    "Example 1:\n"
    "Text: Alice is Bob's mother.\n"
    "Triplets: (Alice, is mother of, Bob)\n"
    "Example 2:\n"
    "Text: Phase III clinical trials are underway for cendakimab in "
    "eosinophilic esophagitis.\n"
    "Triplets:\n"
    "(cendakimab, has indication, Eosinophilic Esophagitis)\n"
    "(cendakimab, has status, Phase III)\n"
    "Example 3:\n"
    "Text: Investigational Compounds\n A/I CELMoD (CC-99282) "
    "(Relapsed/Refractory Non-Hodgkin Lymphoma)\n"
    "Triplets:\n"
    "(CC-99282, has mechanism, CELMoD)\n"
    "(CC-99282, has status, investigational)\n"
    "(CC-99282, has indication, Relapsed/Refractory Non-Hodgkin Lymphoma)\n"
    "Example 4:\n"
    "Text: OPDUALAG (nivolumab + relatlimab) --1L Melanoma\n"
    "Triplets:\n"
    "(OPDUALAG, has synonym, nivolumab + relatlimab)\n"
    "(OPDUALAG, has indication, Melanoma)\n"
    "(OPDUALAG, has indication, 1L Melanoma)\n"
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

BIOMEDICAL_PRODUCT_TRIPLET_EXTRACT_TMPL = (
    "Details about a product are provided below. "
    "Extract related attributes (objects) of interest include:\n"
    " - status: investigational, commercial, LOE, phase, submitted\n"
    " - indications\n"
    " - mechanisms of action\n"
    " - synonyms\n"
    " - probability of success\n"
    "Return a maximum of {max_knowledge_triplets}.\n"
    "Text: {text}\n"
    "Triplets:\n"
)

BIOMEDICAL_PRODUCT_TRIPLET_EXTRACT_PROMPT = KnowledgeGraphPrompt(
    BIOMEDICAL_PRODUCT_TRIPLET_EXTRACT_TMPL
)
