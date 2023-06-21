"""
Training NER models
"""
import spacy

from system import initialize

initialize()

from clients import patent_client
from common.ner.training import train_ner


def main():
    """
    Main
    """
    nlp = spacy.load("en_core_sci_scibert")
    patents = patent_client.search(["melanoma", "asthma"])
    strings = [patent["title"] + "\n" + patent["abstract"] for patent in patents]
    train_ner(nlp, strings)


if __name__ == "__main__":
    main()
