import math
from typing import Sequence, cast
from pydash import flatten, uniq
from spacy.tokens import Doc, Span, Token
import torch
from torch.nn import functional as F
import networkx as nx
from functools import reduce
import logging
from scispacy.candidate_generation import MentionCandidate, KnowledgeBase

from constants.umls import (
    MOST_PREFERRED_UMLS_TYPES,
    PREFERRED_UMLS_TYPES,
    PREFERRED_UMLS_TYPES,
    UMLS_WORD_OVERRIDES,
)
from core.ner.types import CanonicalEntity
from data.domain.biomedical.umls import clean_umls_name, is_umls_suppressed
from utils.list import has_intersection


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

SYNTACTIC_SIMILARITY_WEIGHT = 0.3


def vector_mean(vectors: Sequence[torch.Tensor]) -> torch.Tensor:
    """
    Takes a list of Nx0 vectors and returns the mean vector (Nx0)
    """
    return torch.concat(
        [v.unsqueeze(dim=1) for v in vectors],
        dim=1,
    ).mean(dim=1)


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
        vector_mean([torch.tensor(t.vector) for t in tokens[min(iset) : max(iset)]])
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


def l1_regularize(vector: torch.Tensor) -> torch.Tensor:
    # sparsify
    vector[vector.abs() < 0.3] = 0  # sparsify

    # l1 normalize
    return F.normalize(vector, p=1, dim=0)


def truncated_svd(vector: torch.Tensor, variance_threshold=0.98) -> torch.Tensor:
    """
    Torch implementation of TruncatedSVD
    (from Anthropic)
    """
    # Reshape x if it's a 1D vector
    if vector.ndim == 1:
        vector = vector.unsqueeze(1)

    # l1 reg
    v_sparse = l1_regularize(vector)

    # SVD
    U, S, _ = torch.linalg.svd(v_sparse)

    # Sorted eigenvalues
    E = torch.sort(S**2 / torch.sum(S**2), descending=True)

    # Cumulative energy
    cum_energy = torch.cumsum(E[0], dim=0)

    mask = cum_energy > variance_threshold
    k = torch.sum(mask).int() + 1

    # Compute reduced components
    U_reduced = U[:, :k]

    return cast(torch.Tensor, U_reduced)


def similarity_with_residual_penalty(
    mention_vector: torch.Tensor,
    candidate_vector: torch.Tensor,
    distance: float | None = None,  # cosine/"angular" distance
    alpha: float = 0.5,
) -> float:
    """
    Compute a weighted similarity score that penalizes a large residual.

    Args:
        mention_vector (torch.Tensor): mention vector
        candidate_vector (torch.Tensor): candidate vector
        distance (float, optional): cosine/"angular" distance. Defaults to None, in which case it is computed.
        alpha (float, optional): weight of the residual penalty. Defaults to 0.3.
    """
    if distance is None:
        _distance = F.cosine_similarity(mention_vector, candidate_vector, dim=0)
    else:
        _distance = torch.tensor(distance)

    similarity = 2 - _distance

    # Compute residual
    residual = torch.subtract(mention_vector, candidate_vector)

    # Frobenius norm
    residual_norm = torch.norm(residual, p="fro")

    # Scale residual norm to [0,1] range
    scaled_residual_norm = torch.divide(residual_norm, torch.norm(mention_vector))

    # Weighted score
    score = alpha * similarity + (1 - alpha) * (1 - scaled_residual_norm)

    return score.item()


MIN_ORTHO_DISTANCE = 0.2


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
    candidate_id: str,
    candidate_canonical_name: str,
    candidate_type_ids: Sequence[str],
    candidate_aliases: Sequence[str],
    syntactic_similarity: float | None = None,
) -> float:
    """
    Generate a score for a candidate (semantic or not)

    - suppresses certain CUIs
    - suppresses certain names
    - scores based on
        1. UMLS type (tui)
        2. syntactic similarity (if supplied)

    Args:
        candidate_id (str): candidate ID
        candidate_canonical_name (str): candidate canonical name
        candidate_types (list[str]): candidate types
        candidate_aliases (list[str]): candidate aliases
        syntactic_similarity (float): syntactic similarity score from tfidf vectorizer
            if none, then score is based on type only (used by score_semantic_candidate since it weights syntactic vs semantic similarity)
    """

    if is_umls_suppressed(candidate_id, candidate_canonical_name):
        return 0.0

    def type_score():
        """
        Compute a score based on the UMLS type (tui) of the candidate
        """
        is_preferred_type = has_intersection(
            candidate_type_ids, list(PREFERRED_UMLS_TYPES.keys())
        )

        if not is_preferred_type:
            return 0.8

        is_most_preferred_type = has_intersection(
            candidate_type_ids, list(MOST_PREFERRED_UMLS_TYPES.keys())
        )

        if not is_most_preferred_type:
            return 1.0

        return 1.1

    # give candidates with more aliases a higher score, as proxy for number of ontologies in which it is represented.
    # log base 4 - roughly 10% of entries have 5+ aliases. lower bounds at 0.5
    alias_score = max(
        0.5,
        (math.log(len(candidate_aliases), 4) if len(candidate_aliases) > 1 else 0.0),
    )

    if syntactic_similarity is not None:
        return type_score() * alias_score * syntactic_similarity

    return type_score() * alias_score


def score_semantic_candidate(
    candidate_id: str,
    candidate_canonical_name: str,
    candidate_types: list[str],
    candidate_alises: list[str],
    original_vector: torch.Tensor,
    candidate_vector: torch.Tensor,
    syntactic_similarity: float,
    semantic_distance: float,
) -> float:
    """
    Generate a score for a semantic candidate

    Score based on
    - "score_candidate" rules
    - syntactic similarity


    Args:
        candidate_id (str): candidate ID
        candidate_canonical_name (str): candidate canonical name
        candidate_types (list[str]): candidate types
        syntactic_similarity (float): syntactic similarity score
        original_vector (torch.Tensor): original mention vector
        candidate_vector (torch.Tensor): candidate vector
        semantic_distance (float): semantic distance
    """
    type_score = score_candidate(
        candidate_id, candidate_canonical_name, candidate_types, candidate_alises
    )

    if type_score == 0:
        return 0.0

    semantic_similarity = similarity_with_residual_penalty(
        original_vector, candidate_vector, semantic_distance
    )
    return (
        (1 - SYNTACTIC_SIMILARITY_WEIGHT) * semantic_similarity
        + SYNTACTIC_SIMILARITY_WEIGHT * syntactic_similarity
    ) * type_score


JOIN_PUNCT = ["-", "/", "'"]


def join_punctuated_tokens(doc: Doc) -> list[Token | Span]:
    """
    Join tokens that are separated by punctuation, in certain conditions
    e.g. ['non', '-', 'competitive'] -> "non-competitive"

    Specifically, join tokens separated by punctuation if:
    - the punctuation is '-', '/' or "'"
    - the token before OR after is less than 4 characters

    TODO:
    - this is kinda hacky; we don't robustly know if these things should actually be joined
    - what about free-floating numbers, e.g. "peptide 1"  or "apoe 4"?
    """
    punct_indices = [
        i
        for i, t in enumerate(doc)
        if t.text in JOIN_PUNCT
        and i > 0
        and i < len(doc) - 1
        and (len(doc[i - 1]) < 4 or len(doc[i + 1]) < 4)
    ]
    join_tuples = [(pi - 1, pi, pi + 1) for pi in punct_indices]
    join_indices = flatten(join_tuples)
    join_starts = [j[0] for j in join_tuples]
    tokens: list[Token | Span] = []
    for i in range(len(doc)):
        if i in join_indices:
            if i in join_starts:
                tokens.append(doc[i : i + 3])
            else:
                continue
        else:
            tokens.append(doc[i])

    return tokens


def candidate_to_canonical(
    candidate: MentionCandidate, kb: KnowledgeBase
) -> CanonicalEntity:
    """
    Convert a MentionCandidate to a CanonicalEntity
    """
    # go to kb to get canonical name
    entity = kb.cui_to_entity[candidate.concept_id]
    name = clean_umls_name(
        entity.concept_id,
        entity.canonical_name,
        entity.aliases,
        entity.types,
        False,
    )

    return CanonicalEntity(
        id=entity.concept_id,
        ids=[entity.concept_id],
        name=name,
        aliases=entity.aliases,
        description=entity.definition,
        types=entity.types,
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
                concept_id=overrides[text.lower()],
                aliases=[text],
                similarities=[1],
            )
        ]
    return candidates
