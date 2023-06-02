"""
Debugging functions for NER
"""
from spacy.tokens import Doc


def print_tokens(docs: list[Doc]):
    """
    Prints debug information about tokens
    """
    for token in [token for doc in docs for token in doc]:
        if "inhibitor" in token.lemma_ or "mab" in token.lemma_:
            print(
                token.ent_type_ or "UNKNOWN",
                token.text,
                "tag:",
                token.tag_,
                "pos:",
                token.pos_,
                "dep:",
                token.dep_,
                "lemma:",
                token.lemma_,
                "morph:",
                token.morph,
                "prob:",
                token.prob,
                "head:",
                token.head.text,
                "span:",
                [child for child in token.children],
            )
