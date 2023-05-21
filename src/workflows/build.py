"""
Workflows for building up data
"""

from datetime import datetime

from sources.sec.knowledge_graph import build_knowledge_graph


def main():
    """
    Main
    """
    # PFE, JNJ, NVS (Novartis), RHHBY (Roche), APPV, MRK, Bristol Myers Squibb (BMY)
    start_date = datetime(2020, 1, 1)
    build_knowledge_graph("MRK", start_date)


if __name__ == "__main__":
    main()
