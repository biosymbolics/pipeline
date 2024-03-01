"""
Concept decomposition client
"""

import asyncio
from typing import Sequence
from langchain.output_parsers import ResponseSchema
from pydash import flatten, omit
import logging

from clients.openai.gpt_client import GptApiClient
from core.topic import Topics

from .vector_report_client import VectorReportClient
from .types import SubConcept, VectorSearchParams

RESIDUAL_START_YEAR = 2021

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


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
                description="3-4 paragraphs of technical description of the sub-concept",
                type="string",
            ),
        ]

        self.llm = GptApiClient(schemas=response_schemas)
        self.vector_report_client = VectorReportClient()

    async def decompose_concept(self, concept_description: str) -> list[SubConcept]:
        """
        Decompose a concept into sub-concepts
        """
        prompt = f"""
            What follows is a description of a biomedical concept.

            Please decompose it into 5-10 sub-concepts, which may come in the form of:
            - specific phenotypes of a given disease
            - diseases within a therapeutic area
            - different technologies, strategies or targets for achieving given therapeutic effect
            - and so on.

            Return the answer as an array of json objects with the following fields: name, description.

            The description should be written as if a patent: technical, detailed, precise and making appropriate use of jargon.
            Each description should be three to four paragraphs, standalone and avoid any reference to the other descriptions.
            They should be homogeneous, which is to say having similar specificity, scope and scale relative to the original concept.

            Here is an example:
            name: Galectin-1 antibodies
            description: "Monovalent antibodies such as nanobodies that are specific for galectin-1 are described.
            These monovalent antibodies are able to interfere with the activity of galectin-1,
            and thus may be used for the treatment of diseases associated with dysregulated galectin-1 expression
            and/or activity, such as certain types of cancers, as well as conditions associated with pathological angiogenesis or fibrosis."

            Here is the concept:
            "{concept_description}"
        """

        response = await self.llm.query(prompt, is_array=True)
        sub_concepts = [SubConcept(**r) for r in response]

        return sub_concepts

    async def _generate_subconcept_reports(
        self, sub_concepts: Sequence[SubConcept], skip_ids: list[str] = []
    ) -> list[SubConcept]:
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
        concept_docs_by_year = await asyncio.gather(
            *[
                self.vector_report_client(sc.description, skip_ids=skip_ids)
                for sc in sub_concepts
            ]
        )

        return [
            SubConcept(**omit(sc.model_dump(), "report"), report=report)
            for sc, report in zip(sub_concepts, concept_docs_by_year)
        ]

    async def _generate_residuals(
        self, description: str, sub_concepts: Sequence[SubConcept]
    ) -> list[SubConcept]:
        """
        Generate residual sub-concepts from the original concept
        """

        # get known document ids
        known_ids = flatten([r.ids for sc in sub_concepts for r in sc.report])

        if len(known_ids) == 0:
            raise ValueError("No known ids for residual report")

        residual_docs = await self.vector_report_client.get_top_docs(
            description,
            search_params=VectorSearchParams(
                min_year=RESIDUAL_START_YEAR,
                skip_ids=known_ids,
                # higher threshold for similarity for residuals
                # to avoid just getting what was deemed irrelevant in the original search
                alpha=0.99,
            ),
        )

        topic_map = await Topics.model_topics(
            [d.vector for d in residual_docs],
            [d.description for d in residual_docs],
            existing_labels=[d.name for d in sub_concepts],
            context_strings=[description],
        )

        new_sub_concepts = [SubConcept(**r) for r in topic_map]
        new_reports = await self._generate_subconcept_reports(
            new_sub_concepts, skip_ids=known_ids
        )
        return new_reports

    async def decompose_concept_with_reports(
        self, concept_description: str
    ) -> list[SubConcept]:
        """
        Decompose a concept into sub-concepts and fetch reports for each sub-concept
        """
        sub_concepts = await self.decompose_concept(concept_description)
        sc_reports = await self._generate_subconcept_reports(sub_concepts)

        # makes up silly junk
        # TODO: try LDA or PCA on remaining vectors, using LLM to generate descriptions
        residual_sc_reports = await self._generate_residuals(
            concept_description, sc_reports
        )

        return sc_reports + residual_sc_reports
