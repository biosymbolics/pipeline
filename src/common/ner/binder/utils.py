"""
Utility functions for the Binder NER model
"""

import re


def generate_word_indices(text: str) -> list[tuple[int, int]]:
    """
    Generate word indices for a text

    Args:
        text (str): text to generate word indices for
    """
    word_indices = []
    token_re = re.compile(r"[\s\n]")
    words = token_re.split(text)
    for idx, word in enumerate(words):
        start_char = sum([len(word) + 1 for word in words[:idx]])
        end_char = start_char + len(re.sub("[.,;]$", "", word))
        word_indices.append((start_char, end_char))
    return word_indices
