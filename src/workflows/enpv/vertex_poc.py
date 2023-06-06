"""
Test script for vertex
"""
from datetime import datetime
from google.cloud import aiplatform

from clients.llama_index.indices.knowledge_graph import get_kg_index
from clients.llama_index.visualization import (
    visualize_network_by_index,
    list_triples_by_index,
)
from sources.sec.sec import fetch_annual_reports_with_sections as fetch_annual_reports


def main():
    """
    Main
    """
    aiplatform.init(project="fair-abbey-386416")
    section_map = fetch_annual_reports("LLY", datetime(2022, 1, 1))

    # this is the slow part
    for period, sections in section_map.items():
        index = get_kg_index(
            namespace="VERTEX_TEST",
            index_id=period,
            documents=sections,
            model_name="VertexAI",
        )
        list_triples_by_index(index)
        visualize_network_by_index(index)


if __name__ == "__main__":
    main()
