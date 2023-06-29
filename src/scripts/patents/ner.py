"""
This script applies NER to the patent dataset and saves the results to a temporary location.
"""
import sys

from system import initialize

initialize()

from clients.patents.enrich import enrich_with_ner

if __name__ == "__main__":
    if "-h" in sys.argv:
        print(
            "Usage: python3 ner.py\nLoads NER data for patents and saves it to a temporary location"
        )
        sys.exit()
    enrich_with_ner(
        ["asthma", "schizophrenia", "pulmonary hypertension", "bipolar disorder"]
    )
