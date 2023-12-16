from typing import Sequence, cast
from pydash import flatten, uniq
from spacy.tokens import Token
import torch
import torch.nn.functional as F
import networkx as nx
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
        )
        .mean(dim=1)
        .tolist()
        for iset in index_sets
    ]
    return list(zip(ngrams, vectors))


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


LAMBDA_1 = 0.1


def l1_normalize(vector: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.abs(vector) - LAMBDA_1, min=0)


def truncated_svd(vector: torch.Tensor, variance_threshold=0.98) -> torch.Tensor:
    """
    Torch implementation of TruncatedSVD
    (from Anthropic)
    """
    # Reshape x if it's a 1D vector
    if vector.ndim == 1:
        vector = vector.unsqueeze(1)

    # l1 normalization
    v_sparse = l1_normalize(vector)

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
    similarity: float,
    alpha: float = 0.3,
) -> float:
    """
    Compute a weighted similarity score that penalizes a large residual.
    """
    # Compute residual
    residual = mention_vector - candidate_vector

    # Frobenius norm
    residual_norm = torch.norm(residual, p="fro")

    # Scale residual norm to [0,1] range
    scaled_residual_norm = residual_norm / torch.norm(mention_vector)

    # Weighted score
    score = alpha * similarity + (1 - alpha) * (1 - scaled_residual_norm)

    return score.item()


def whiten(X: torch.Tensor) -> torch.Tensor:
    if X.ndim == 1:
        X = X.unsqueeze(1)

    X = X - torch.mean(X, 0)
    Xcov = torch.mm(X.t(), X) / (X.shape[0] - 1)  # Compute  Covariance Matrix
    U, S, V = torch.svd(Xcov)  # Singular Value Decomposition
    D = torch.diag(1.0 / torch.sqrt(S + 1e-5))  # Build Whiten matrix
    W = torch.chain_matmul(U, D, U.t())  # Whitening Matrix
    X_white = torch.mm(X, W)  # Whitened Data

    return X_white.squeeze()


def get_orthogonal_members(mem_vectors: torch.Tensor) -> list[int]:
    """
    Get the members of a composite that are orthogonal to each other
    """

    # Compute pairwise cosine sim
    # X = F.normalize(mem_vectors, p=2, dim=1)
    # dot_products = X @ X.t()
    # sim = (1 + dot_products) / 2

    dist = torch.cdist(mem_vectors, mem_vectors, p=2)
    max_dist = torch.dist(
        torch.zeros_like(mem_vectors), torch.ones_like(mem_vectors), p=2
    )
    scaled_dist = dist / max_dist
    print(scaled_dist)

    edge_index = (scaled_dist < 0.50).nonzero()

    # Connected components clusters similar cand groups
    G = nx.from_edgelist(edge_index.tolist())
    components = nx.connected_components(G)

    c_list = [list(c) for c in components if len(c) > 1]

    removal_indicies = flatten([c[1:] for c in c_list])

    print("remove", removal_indicies)

    return [i for i in range(mem_vectors.size(0)) if i not in uniq(removal_indicies)]
