"""
Training NER models
"""
import spacy

from system import initialize

initialize()

from clients import patent_client
from common.ner.training import weakly_train_ner


def main():
    """
    Main

    After running this, train the model using:
    ```
    spacy init config - --lang en --pipeline ner --optimize accuracy |
    spacy train - --paths.train ./data/patent-sci.spacy  --paths.dev ./data/patent-sci.spacy
    --initialize.vectors en_core_sci_scibert --output ./data/patent-sci
    ```
    """
    nlp = spacy.load("en_core_sci_scibert")
    patents = patent_client.search(["melanoma", "asthma"])
    strings = [patent["title"] + "\n" + patent["abstract"] for patent in patents]
    weakly_train_ner(nlp, strings, "../data/patent-sci.spacy")


if __name__ == "__main__":
    main()
