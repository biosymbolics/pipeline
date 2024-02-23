"""
Concept decomposition client
"""

import asyncio
from typing import Sequence
from langchain.output_parsers import ResponseSchema

from clients.openai.gpt_client import GptApiClient
from core.ner.spacy import get_transformer_nlp

from .semantic_finder import VectorReportClient
from .types import SubConcept, TopDocsByYear


class ConceptDecomposer:
    """
    Class for trend decomposition client

    - Accepts a query describing a general concept (e.g. drug delivery systems to increase BBB penetration)
    - Decomposes the trend into sub-concepts (e.g. efflux pump inhibitors vs receptor-mediated transport vs intranasal delivery)
    - Finds documents related to sub-concepts via vector similarity
    - Returns summary information
    """

    def __init__(self):
        response_schemas = [
            ResponseSchema(
                name="name", description="short sub-concept name", type="string"
            ),
            ResponseSchema(
                name="description",
                description="2-4 paragraph description of the sub-concept",
                type="string",
            ),
        ]

        self.llm = GptApiClient(schemas=response_schemas, model="gpt-4")
        self.nlp = get_transformer_nlp()
        self.vector_report_client = VectorReportClient()

    async def decompose_concept(self, concept_description: str) -> list[SubConcept]:
        """
        Decompose a concept into sub-concepts
        """
        prompt = f"""
            What follows is a description of a biomedical research concept:
            {concept_description}

            Please decompose this concept into sub-concepts, which may come in the form of:
            - specific phenotypes of a given disease
            - diseases within a therapeutic area
            - different ways of achieving a given therapeutic effect

            Return the answer as an array of json objects with the following fields: name, description.
            The description should be technical and detailed, non-self-referential, and 2-4 paragraphs in length.
        """

        response = await self.llm.query(prompt, is_array=True)
        sub_concepts = [SubConcept(**r) for r in response]

        return sub_concepts

    async def _generate_subconcept_reports(
        self, sub_concepts: Sequence[SubConcept]
    ) -> dict[str, list[TopDocsByYear]]:
        """
        Fetches reports for each sub-concept

        Returns a dictionary with sub-concept names as keys and lists of TopDocsByYear as values
        Example:
        {
            "efflux pump inhibitors": [{ year: 2011, avg_score: 0.43, ... }, ...],
            "receptor-mediated transport": [{ year: 2011, avg_score: 0.31, ... }, ...],
            "intranasal delivery": [{ year: 2011, avg_score: 0.57, ... }, ...],
        }
        """
        sub_concept_names = [sc.name for sc in sub_concepts]
        concept_docs_by_year = await asyncio.gather(
            *[self.vector_report_client(sc.description) for sc in sub_concepts]
        )

        return dict(zip(sub_concept_names, concept_docs_by_year))

    async def decompose_concept_with_reports(
        self, concept_description: str
    ) -> dict[str, list[TopDocsByYear]]:
        """
        Decompose a concept into sub-concepts and fetch reports for each sub-concept
        """
        sub_concepts = await self.decompose_concept(concept_description)
        return await self._generate_subconcept_reports(sub_concepts)
