import re
from typing import Sequence
from pydash import flatten, uniq
from spacy.tokens import Token
import torch
import networkx as nx
from functools import reduce
import logging

from constants.umls import (
    CANDIDATE_TYPE_WEIGHT_MAP,
    UMLS_WORD_OVERRIDES,
)
from core.ner.types import CanonicalEntity
from data.domain.biomedical.umls import clean_umls_name, is_umls_suppressed
from typings.documents.common import MentionCandidate
from utils.re import get_or_re
from utils.tensor import similarity_with_residual_penalty, tensor_mean


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SYNTACTIC_SIMILARITY_WEIGHT = 0.3
MIN_ORTHO_DISTANCE = 0.2


def generate_ngrams(
    tokens: Sequence[Token], n: int
) -> list[tuple[tuple[str, ...], torch.Tensor]]:
    """
    Generate n-grams (term & token) from a list of Spacy tokens

    Args:
        tokens (Sequence[Token]): list of tokens
        n (int): n-gram size

    Returns:
        list[tuple[tuple[str, str], torch.Tensor]]: list of n-grams tuples and their vectors
    """
    if n < 2:
        raise ValueError("n must be >= 2")

    index_sets: list[tuple[int, ...]] = reduce(
        lambda acc, i: acc + [(i, *[i + grm + 1 for grm in range(n - 1)])],
        range(len(tokens) + 1 - n),
        [],
    )
    ngrams = [tuple(tokens[i].text for i in iset) for iset in index_sets]
    vectors = [
        tensor_mean([torch.tensor(t.vector) for t in tokens[min(iset) : max(iset)]])
        for iset in index_sets
    ]
    return list(zip(ngrams, vectors))


def generate_ngram_phrases(
    tokens: Sequence[Token], n: int
) -> list[tuple[str, torch.Tensor]]:
    """
    Generate n-grams (term & token) from a list of Spacy tokens

    Args:
        tokens (Sequence[Token]): list of tokens
        n (int): n-gram size

    Returns:
        list[tuple[str, list[float]]]: list of n-grams and their vectors
    """
    return [(" ".join(ng[0]), ng[1]) for ng in generate_ngrams(tokens, n)]


def get_orthogonal_members(
    mem_vectors: torch.Tensor, min_ortho_distance: float = MIN_ORTHO_DISTANCE
) -> list[int]:
    """
    Get the members of a composite that are semantically orthogonal-ish to each other
    """
    dist = torch.cdist(mem_vectors, mem_vectors, p=2)
    max_dist = torch.dist(
        torch.zeros_like(mem_vectors), torch.ones_like(mem_vectors), p=2
    )
    scaled_dist = dist / max_dist
    edge_index = (scaled_dist < min_ortho_distance).nonzero()

    # Connected components clusters similar cand groups
    G = nx.from_edgelist(edge_index.tolist())
    components = nx.connected_components(G)

    c_list = [list(c) for c in components if len(c) > 1]
    removal_indicies = flatten([c[1:] for c in c_list])

    logger.info("Removing non-ortho indices: %s (%s)", removal_indicies, scaled_dist)

    return [i for i in range(mem_vectors.size(0)) if i not in uniq(removal_indicies)]


def score_candidate(
    candidate: MentionCandidate,
    mention_vector: torch.Tensor,
    is_composite: bool,
) -> float:
    """
    Generate a score for a semantic candidate

    - suppresses certain CUIs
    - suppresses certain names
    - scores based on
        1. UMLS type (tui)
        2. syntactic similarity
        3. semantic similarity (with residual penalty) similarity
    """
    if is_umls_suppressed(candidate.id, candidate.name, [], is_composite):
        return 0.0

    # give candidates with more aliases a higher score, as proxy for num. ontologies in which it is represented.
    alias_score = 1.1 if len(candidate.synonyms) >= 8 else 1.0

    # score based on the UMLS type (tui) of the candidate
    type_score = max([CANDIDATE_TYPE_WEIGHT_MAP.get(ct, 0.7) for ct in candidate.types])

    base_score = round(type_score * alias_score, 3)

    if base_score == 0:
        return 0.0

    semantic_similarity = similarity_with_residual_penalty(
        mention_vector, torch.tensor(candidate.vector), name=candidate.name
    )
    return (
        (1 - SYNTACTIC_SIMILARITY_WEIGHT) * semantic_similarity
        + SYNTACTIC_SIMILARITY_WEIGHT * candidate.syntactic_similarity
    ) * base_score


def candidate_to_canonical(candidate: MentionCandidate) -> CanonicalEntity:
    """
    Convert a MentionCandidate to a CanonicalEntity
    """
    # go to kb to get canonical name
    name = clean_umls_name(
        candidate.id,
        candidate.name,
        candidate.synonyms,
        candidate.types,
        False,
    )

    return CanonicalEntity(
        id=candidate.id,
        ids=[candidate.id],
        name=name,
        aliases=candidate.synonyms,
        description="",
        types=candidate.types,
    )


def apply_umls_word_overrides(
    text: str,
    candidates: list[MentionCandidate],
    overrides: dict[str, str] = UMLS_WORD_OVERRIDES,
) -> list[MentionCandidate]:
    """
    Certain words we match to an explicit cui (e.g. "modulator" -> "C0005525")
    """
    # look for any overrides (terms -> candidate)
    has_override = text.lower() in overrides
    if has_override:
        return [
            MentionCandidate(
                **{
                    k: v
                    for k, v in candidates[0].model_dump().items()
                    if k not in ["id", "synonyms", "semantic_similarity"]
                },
                id=overrides[text.lower()],
                synonyms=[text],
                semantic_similarity=1,
            )
        ]
    return candidates


MATCH_RETRY_REWRITES = {
    # Rewrite a dash to:
    # 1) empty string if the next word is less than 3 character, e.g. tnf-a -> tnfa
    # 2) space if the next word is >= 3 characters, e.g. tnf-alpha -> tnf alpha
    # Reason: tfidf vectorizer and/or the UMLS data have inconsistent handling of hyphens
    r"(\w{2,})-(\w{1,3})": r"\1\2",
    r"(\w)-(\w{1,3})": r"\1 \2",
    r"(\w+)-(\w{4,})": r"\1 \2",
    "inhibitor": "antagonist",
    "antagonist": "inhibitor",
    "agent": "",
    "drug": "",
    "receptor": "",
    "anti": "",
    "antibod(?:y|ie)": "",
    "ligand": "",
    "targeting": "",
    "penetration": "",
    "expression": "",
    "diseases?": "disorder",
    r"\s{2,}": " ",
}

MATCH_RETRY_RE = get_or_re(
    list(MATCH_RETRY_REWRITES.keys()), enforce_word_boundaries=True
)


def apply_match_retry_rewrites(text: str) -> str | None:
    """
    Rewrite text before retrying a match

    Returns None if no rewrite is needed
    """
    new_text = text

    # look for any overrides (terms -> candidate)
    match = re.search(MATCH_RETRY_RE, text, re.IGNORECASE)

    if match is not None:
        for pattern, repl in MATCH_RETRY_REWRITES.items():
            new_text = re.sub(rf"\b{pattern}s?\b", repl, new_text, flags=re.IGNORECASE)

    if new_text != text:
        return new_text

    return None
