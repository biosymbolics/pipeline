"""
LLM prompts for SEC doc parsing
TODO:
- turn the schema/validation/prompt thing into reusable method
- train LLM on validation if it tends to be a problem (or fix)
"""

from llama_index.prompts.prompts import KnowledgeGraphPrompt

JSON_PIPELINE_SCHEMA = {
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
    "Some text is provided below. Given the text, extract up to {max_knowledge_triplets} "
    "of the most relevant knowledge triplets in the form of (subject, predicate, object) "
    "about in which the subject or predicate is an intervention (compounds, mechanisms of action or drug class). "
    "Avoid stopwords.\n"
    "---------------------\n"
    "Example 1:\n"
    "Text: Investigational Compounds\n"
    "alnuctamab BCMA TCE --Relapsed/Refractory Multiple Myeloma"
    "Anti-SIRPα --Hematologic Malignancies \n"
    "Triplets:\n"
    "(alnuctamab, has indication, Relapsed/Refractory Multiple Myeloma)\n"
    "(Anti-SIRPα, has indication, Hematologic Malignancies)\n"
    "(alnuctamab, is an, investigational compound)\n"
    "(Anti-SIRPα, is an, investigational mechanism of action)\n"
    "Example 2:\n"
    "Text: For immunology, the Phase III clinical trials are underway for cendakimab in eosinophilic esophagitis.\n"
    "Triplets:\n"
    "(cendakimab, has indication, Eosinophilic Esophagitis)\n"
    "(cendakimab, is an, investigational compound)\n"
    "(cendakimab, is in, Phase III)\n"
    "Example 3:\n"
    "Text: Investigational Compounds\n"
    "A/I CELMoD (CC-99282) (Relapsed/Refractory Non-Hodgkin Lymphoma)"
    "Triplets:\n"
    "(CC-99282, has mechanism, CELMoD)\n"
    "(CC-99282, is an, investigational compound)\n"
    "(CC-99282, has indication, Relapsed/Refractory Non-Hodgkin Lymphoma)\n"
    "Example 4:\n"
    "Text: We continue to advance the next wave of innovative medicines by investing "
    "significantly in our oncology, hematology (with alnuctamab in multiple myeloma), "
    "immunology (with LPA1 antagonist in pulmonary fibrosis).\n"
    "Triplets:\n"
    "(LPA1 Antagonist, has indication, Pulmonary Fibrosis)\n"
    "(alnuctamab, has indication, Multiple Myeloma)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)

BIOMEDICAL_TRIPLET_EXTRACT_PROMPT = KnowledgeGraphPrompt(
    BIOMEDICAL_TRIPLET_EXTRACT_TMPL
)