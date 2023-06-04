"""
Functions specific to knowledge graph indices
"""
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
        name="details_from_doc",
        description="all other information about this intervention, extracted from the document",
    ),
    ResponseSchema(
        name="details", description="everything else you know about this intervention"
    ),
]


def create_entity_index(
    entity: str, vector_index: GPTVectorStoreIndex, namespace: str, index_id: str
) -> GPTVectorStoreIndex:
    """
    Summarize entity based on the document and persist in an index

    Args:
        entity (str): entity name
        vector_index (GPTVectorStoreIndex): vector index to use for lookups
        namespace (str): namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
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

    about_entity = query_index(
        vector_index, query, prompt=qa_prompt, refine_prompt=refine_prompt
    )
    logging.info("about entity %s: %s", entity, about_entity)

    try:
        entity_obj = output_parser.parse(about_entity)
    except Exception as ex:
        logging.error("Could not parse entity %s: %s", entity, ex)
        raise ex

    name = entity_obj.get("name") or entity
    details = entity_obj.get("details")
    index = get_vector_index("entities", index_id + f"{namespace}-{name}", [details])
    return index


def get_entity_indices(
    entities: list[str], namespace: str, index_id: str, documents: list[str]
) -> list[GPTVectorStoreIndex]:
    """
    For each entity in the provided list, summarize based on the document and persist in an index

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
