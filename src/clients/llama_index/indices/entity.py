"""
Functions specific to knowledge graph indices
"""
from typing import Optional
from llama_index import GPTVectorStoreIndex
from llama_index.output_parsers import LangchainOutputParser
from llama_index.prompts.prompts import QuestionAnswerPrompt, RefinePrompt
from llama_index.prompts.default_prompts import (
    DEFAULT_TEXT_QA_PROMPT_TMPL,
    DEFAULT_REFINE_PROMPT_TMPL,
)
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
import logging
from pydash import compact

from clients.llama_index.indices.vector import get_vector_index
from types.indices import NamespaceKey
from clients.llama_index.utils import get_namespace
from common.utils.string import get_id
from sources.sec.prompts import GET_BIOMEDICAL_NER_TEMPLATE
from .general import query_index

response_schemas = [
    ResponseSchema(name="name", description="normalized intervention name"),
    ResponseSchema(name="details", description="details about this intervention"),
]


def create_entity_index(
    entity: str,
    vector_index: GPTVectorStoreIndex,
    namespace_key: NamespaceKey,
    index_id: str,
) -> Optional[GPTVectorStoreIndex]:
    """
    Summarize entity based on the document and persist in an index

    Args:
        entity (str): entity name
        vector_index (GPTVectorStoreIndex): vector index to use for lookups
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
    """
    prompt = GET_BIOMEDICAL_NER_TEMPLATE(entity)

    lc_output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    output_parser = LangchainOutputParser(lc_output_parser)
    fmt_qa_tmpl = output_parser.format(DEFAULT_TEXT_QA_PROMPT_TMPL)
    fmt_refine_tmpl = output_parser.format(DEFAULT_REFINE_PROMPT_TMPL)
    qa_prompt = QuestionAnswerPrompt(fmt_qa_tmpl, output_parser=output_parser)
    refine_prompt = RefinePrompt(fmt_refine_tmpl, output_parser=output_parser)

    about_entity = query_index(
        vector_index, prompt, prompt=qa_prompt, refine_prompt=refine_prompt
    )
    logging.info("about entity %s: %s", entity, about_entity)

    try:
        entity_obj = output_parser.parse(about_entity)
        name = entity_obj.get("name") or entity
        details = entity_obj.get("details")
        namespace = get_namespace(namespace_key)
        index = get_vector_index(
            ("entities",), index_id + f"{namespace}-{get_id(name)}", [details]
        )
        return index
    except Exception as ex:
        logging.error("Could not parse entity %s: %s", entity, ex)

    return None


def get_entity_indices(
    entities: list[str],
    namespace_key: NamespaceKey,
    index_id: str,
    documents: list[str],
) -> list[GPTVectorStoreIndex]:
    """
    For each entity in the provided list, summarize based on the document and persist in an index

    Args:
        entities (list[str]): list of entities to get indices for
        namespace_key (NamespaceKey) namespace of the index (e.g. SEC-BMY)
        index_id (str): unique id of the index (e.g. 2020-01-1)
        documents (Document): list of llama_index Documents
    """
    index = get_vector_index(namespace_key, index_id, documents)
    indices = [
        create_entity_index(entity, index, namespace_key, index_id)
        for entity in entities
    ]
    return compact(indices)
