"""
LLM prompts for SEC doc parsing
TODO:
- turn the schema/validation/prompt thing into reusable method
- train LLM on validation if it tends to be a problem (or fix)
"""

from typing import Mapping, TypedDict
from llama_index.prompts.prompts import KnowledgeGraphPrompt


class Schema(TypedDict):
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
    "Section 1 of a SEC 10-K document is provided below in HTML. Given this content, "
    "extract knowledge about all the company's products as triplets (subject, predicate, object) "
    "(up to {max_knowledge_triplets}).\n"
    "Extract ALL products (aka compounds) and related attributes such as:\n"
    " - status (investigational, commercial, LOE, Phase I/II/III, submitted, Fast Track)\n"
    " - indication(s)\n"
    " - mechanism(s) of action\n"
    " - target(s)\n"
    "Avoid stop words and long predicates. "
    "Triplet order matters, so be consistent, e.g. ('ABC Gene Therapy', 'has status', 'Phase II'). "
    "Note that products are often found in html tables or lists.\n"
    "---------------------\n"
    "Example 1:\n"
    "Text: Alice is Bob's mother.\n"
    "Triplets: (Alice, is mother of, Bob)"
    "Example 2:\n"
    "Text: Investigational Compounds\n"
    "Anti-SIRPα Hematologic Malignancies\n"
    "Triplets:\n"
    "(Anti-SIRPα, has indication, Hematologic Malignancies)\n"
    "(Anti-SIRPα, has status, investigational)\n"
    "Example 3:\n"
    "Text: For immunology, the Phase III clinical trials are underway for cendakimab in eosinophilic esophagitis.\n"
    "Triplets:\n"
    "(cendakimab, has indication, Eosinophilic Esophagitis)\n"
    "(cendakimab, has status, investigational)\n"
    "(cendakimab, has status, Phase III)\n"
    "Example 4:\n"
    "Text: Investigational Compounds\n"
    "A/I CELMoD (CC-99282) (Relapsed/Refractory Non-Hodgkin Lymphoma)"
    "Triplets:\n"
    "(CC-99282, has mechanism, CELMoD)\n"
    "(CC-99282, has status, investigational)\n"
    "(CC-99282, has indication, Relapsed/Refractory Non-Hodgkin Lymphoma)\n"
    "Example 5:\n"
    "Text: OPDUALAG (fixed dose nivolumab + relatlimab) --1L Melanoma\n"
    "Triplets:\n"
    "(OPDUALAG, contains, nivolumab)\n"
    "(OPDUALAG, contains, relatlimab)\n"
    "(OPDUALAG, has indication, Melanoma)\n"
    "(OPDUALAG, has indication, 1L Melanoma)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)

BIOMEDICAL_TRIPLET_EXTRACT_PROMPT = KnowledgeGraphPrompt(
    BIOMEDICAL_TRIPLET_EXTRACT_TMPL
)
