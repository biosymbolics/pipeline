"""
LLM prompts for SEC doc parsing
TODO:
- turn the schema/validation/prompt thing into reusable method
- train LLM on validation if it tends to be a problem (or fix)
"""

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
    "What are the products in this pharma company's R&D pipeline? "
    "Return results as an array of json objects containing "
    "the keys " + ", ".join(JSON_PIPELINE_SCHEMA["properties"].keys()) + "."
)
