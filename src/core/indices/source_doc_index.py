"""
SourceDocIndex
"""
from datetime import datetime
from typing import Any, Optional
from llama_index import GPTVectorStoreIndex
from langchain.output_parsers import ResponseSchema

from clients.llama_index import upsert_index, get_index, query_index
from clients.llama_index.context import ContextArgs, DEFAULT_CONTEXT_ARGS
from clients.llama_index.parsing import get_prompts_and_parser, parse_answer
from clients.llama_index.types import DocMetadata
from clients.vector_dbs.pinecone import get_metadata_filters
from common.utils.namespace import get_namespace_id
from typings.indices import LlmIndex, NamespaceKey, Prompt, RefinePrompt

INDEX_NAME = "source-docs"


class SourceDocIndex:
    """
    SourceDocIndex

    Simple index over raw-ish source docs

    Pinecone query example:
    ``` json
    { "$and": [{ "company": "PFE" }, { "doc_source": "SEC" }, { "doc_type": "10-K" }, { "period": "2020-12-31" }] }
    ```

    """

    def __init__(
        self,
        context_args: ContextArgs = DEFAULT_CONTEXT_ARGS,
        index_impl: LlmIndex = GPTVectorStoreIndex,  # type: ignore
    ):
        """
        initialize

        Args:
            context_args (ContextArgs, optional): context args. Defaults to DEFAULT_CONTEXT_ARGS.
            index_impl (LlmIndex, optional): index implementation. Defaults to GPTVectorStoreIndex.
        """
        self.context_args = context_args
        self.index = None
        self.index_impl = index_impl

        self.__load()

    def add_documents(
        self,
        source: NamespaceKey,
        documents: list[str],
        retrieval_date: datetime = datetime.now(),
    ):
        """
        Load docs into index with metadata

        Args:
            documents (list[str]): list of documents
            source (NamespaceKey): source namespace, e.g.
                ``` python
                dict_to_named_tuple(
                    {
                        "company": "PFE",
                        "doc_source": "SEC",
                        "doc_type": "10-K",
                        "period": "2020-12-31",
                    }
                )
                ```
            retrieval_date (datetime, optional): retrieval date of source docs. Defaults to datetime.now().
        """

        def __get_metadata(doc) -> DocMetadata:
            return {
                **source._asdict(),
                # "retrieval_date": retrieval_date.isoformat(),
            }

        # uniq doc id for deduplication/idempotency
        def __get_doc_id(doc) -> str:
            return get_namespace_id(source)

        index = upsert_index(
            INDEX_NAME,
            documents,
            index_impl=self.index_impl,
            get_doc_metadata=__get_metadata,
            get_doc_id=__get_doc_id,
            context_args=self.context_args,
        )
        self.index = index

    def __load(self):
        """
        Load source doc index from disk
        """
        index = get_index(INDEX_NAME, **self.context_args.storage_args)
        self.index = index

    def query(
        self,
        query_string: str,
        source: NamespaceKey,
        prompt_template: Optional[Prompt] = None,
        refine_prompt: Optional[RefinePrompt] = None,
    ) -> Any:
        """
        Query the index

        Args:
            query_string (str): query string
            source (NamespaceKey): source namespace that acts as filter, e.g.
                ``` python
                dict_to_named_tuple(
                    {
                        "company": "BIBB",
                        "doc_source": "SEC",
                        "doc_type": "10-K",
                        "period": "2020-12-31",
                    }
                )
                ```
            prompt_template (Prompt, optional): prompt. Defaults to None.
            refine_prompt (RefinePrompt, optional): refine prompt. Defaults to None.
        """
        if not self.index:
            raise ValueError("Index not initialized.")

        metadata_filters = get_metadata_filters(source)

        answer = query_index(
            self.index,
            query_string,
            prompt_template=prompt_template,
            refine_prompt=refine_prompt,
            metadata_filters=metadata_filters,
        )

        return answer

    def query_for_entities(self, source: NamespaceKey) -> list[str]:
        """
        Query for entities
        """
        prompt = f"""
            What compounds, drugs and MoAs is this company working on?
        """
        response_schemas = [
            ResponseSchema(
                name="products",
                description="all products as a string array, e.g. ['drug1', 'drug2']",
            ),
        ]

        prompts, parser = get_prompts_and_parser(response_schemas)
        response = self.query(prompt, source, *prompts)

        result = parse_answer(response, parser, return_orig_on_fail=False)

        return result["products"]

    def confirm_entities(
        self, possible_entities: list[str], source: NamespaceKey
    ) -> list[str]:
        """
        Confirm a list of entities
        """
        prompt = f"""
            Using NER (named entity recognition), we found a list of interventions in the source document.
            However, we are not sure if they are correct.
            There may be false positives (not actually intervention names, e.g. "collaborations"),
            entries that are too generic (e.g. "anti-infective products"),
            entries that contain too much additional information (e.g. "oral anti-coagulant market share gains"),
            and missing interventions (entities not recognized by the NER model).
            Please confirm the list of interventions, correcting, removing and adding as necessary.
            Deduplicate before returning the list. The list of NER-detected interventions is:
            {possible_entities}.
        """
        response_schemas = [
            ResponseSchema(
                name="products",
                description="all products as a string array, e.g. ['drug1', 'drug2']",
            ),
        ]

        prompts, parser = get_prompts_and_parser(response_schemas)
        response = self.query(prompt, source, *prompts)

        result = parse_answer(response, parser, return_orig_on_fail=False)

        if "products" not in result:
            raise ValueError("No products found in result %s", result)

        return result["products"]
