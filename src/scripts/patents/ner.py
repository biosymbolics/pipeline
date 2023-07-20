"""
This script applies NER to the patent dataset and saves the results to a temporary location.
"""
import sys

from system import initialize

initialize()

from clients.patents.extract_entities import PatentEnricher

if __name__ == "__main__":
    if "-h" in sys.argv:
        print("Usage: python3 ner.py\nLoads NER data for patents")
        sys.exit()
    enricher = PatentEnricher()
    terms = [
        "schizophrenia",
        "pulmonary hypertension",
        "bipolar disorder",
        "depression",
        "major depressive disorder",
        "asthma",
        "melanoma",
        "alzheimer's disease",
    ]
    enricher()
