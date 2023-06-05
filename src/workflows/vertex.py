"""
Test script for vertex
"""
from datetime import datetime
from google.cloud import aiplatform

from clients.llama_index.indices.knowledge_graph import create_and_query_kg_index
from sources.sec.sec import fetch_annual_reports_with_sections as fetch_annual_reports


def main():
    """
    Main
    """
    aiplatform.init(project="fair-abbey-386416")
    section_map = fetch_annual_reports("LLY", datetime(2022, 1, 1))

    # this is the slow part
    for period, sections in section_map.items():
        answer = create_and_query_kg_index(
            query="What products is this pharma company developing?",
            namespace="VERTEX_TEST",
            index_key=period,
            documents=sections,
            model_name="VertexAI",
        )
        return answer


if __name__ == "__main__":
    main()
