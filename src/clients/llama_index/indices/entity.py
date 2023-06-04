"""
Functions specific to knowledge graph indices
"""
import json
from llama_index import GPTVectorStoreIndex
from llama_index.output_parsers import LangchainOutputParser
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_REFINE_PROMPT_TMPL,
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import logging

from clients.llama_index.indices.vector import get_vector_index
from .general import query_index

response_schemas = [
    ResponseSchema(name="name", description="normalized drug or MoA name"),
    ResponseSchema(
        name="details", description="all other information about this intervention"
    ),
]


def create_entity_index(
    entity: str, vector_index: GPTVectorStoreIndex, namespace: str, index_id: str
) -> GPTVectorStoreIndex:
    """
    Get (or create) the knowledge graph index for a single entity

    Args:
        entity (str): entity name
        vector_index (GPTVectorStoreIndex): vector index to use for lookups
    """
    query = (
        f"Is {entity} a pharmaceutical compound or mechanism of action? "
        "If yes, please return information about this drug, such as: "
        "drug class, mechanism of action, indication(s), status, and clinical trials. "
        "If the drug is commercial, please include details about the revenue it generates, competition and prospects for the future. "
        "If the drug is investigational, please include details about its phase of development and probability of success. "
    )

    lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = LangchainOutputParser(lc_output_parser)
    fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
    fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
    qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
    refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)

    about_entity = json.loads(
        query_index(vector_index, query, prompt=qa_prompt, refine_prompt=refine_prompt)
    )
    logging.info("about entity %s: %s", entity, about_entity)

    name = about_entity.get("name") or entity
    details = about_entity.get("details")
    index = get_vector_index("entities", index_id + f"{namespace}-{name}", [details])
    return index


def get_entity_indices(
    entities: list[str], namespace: str, index_id: str, documents: list[str]
):
    """
    For each entity in the provided list, get (or create) the knowledge graph index

    Args:
        entities (list[str]): list of entities to get indices for
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
    """
    index = get_vector_index(namespace, index_id, documents)
    indices = [
        create_entity_index(entity, index, namespace, index_id) for entity in entities
    ]
    return indices
