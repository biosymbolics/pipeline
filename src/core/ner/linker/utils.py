from typing import Sequence
import torch
from spacy.tokens import Token
import torch
from functools import reduce


def generate_ngrams(
    tokens: Sequence[Token], n: int
) -> list[tuple[tuple[str, ...], list[float]]]:
    """
    Generate n-grams (term & token) from a list of Spacy tokens

    Args:
        tokens (Sequence[Token]): list of tokens
        n (int): n-gram size

    Returns:
        list[tuple[tuple[str, str], list[float]]]: list of n-grams tuples and their vectors
    """
    index_sets: list[tuple[int, ...]] = reduce(
        lambda acc, i: acc + [(i, *[i + grm + 1 for grm in range(n - 1)])],
        range(len(tokens) + 1 - n),
        [],
    )
    ngrams = [tuple(tokens[i].text for i in iset) for iset in index_sets]
    vectors = [
        torch.concat(
            [
                torch.tensor(t.vector).unsqueeze(dim=1)
                for t in tokens[min(iset) : max(iset)]
            ],
            dim=1,
        ).mean(dim=1)
        # .tolist()
        for iset in index_sets
    ]
    print("SHSHS", [v.shape for v in vectors])
    return list(zip(ngrams, [v.tolist() for v in vectors]))


def generate_ngram_phrases(
    tokens: Sequence[Token], n: int
) -> list[tuple[str, list[float]]]:
    """
    Generate n-grams (term & token) from a list of Spacy tokens

    Args:
        tokens (Sequence[Token]): list of tokens
        n (int): n-gram size

    Returns:
        list[tuple[str, list[float]]]: list of n-grams and their vectors
    """
    return [(" ".join(ng[0]), ng[1]) for ng in generate_ngrams(tokens, n)]
