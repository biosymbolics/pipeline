"""
Concept decomposition client
"""

import asyncio
from typing import Sequence
from langchain.output_parsers import ResponseSchema
from pydantic import BaseModel

from clients.openai.gpt_client import GptApiClient
from clients.vector.company_finder import SemanticCompanyFinder
from core.ner.spacy import get_transformer_nlp
from typings.client import CompanyFinderParams


class SubConcept(BaseModel):
    name: str
    description: str


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
        self.semantic_client = SemanticCompanyFinder()

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

    async def fetch_subconcept_details(self, sub_concepts: Sequence[SubConcept]):
        """
        Fetches details for sub-concepts
        """

        async def fetch_detail(description: str):
            results = await self.semantic_client(
                CompanyFinderParams(description=description)
            )

        details = await asyncio.gather(
            *[fetch_detail(sc.description) for sc in sub_concepts]
        )
