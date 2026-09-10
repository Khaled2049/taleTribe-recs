"""Maximal Marginal Relevance: the last step before results are returned.

Pure relevance ranking produces a shelf of near-identical books. Ask for
something like Dune and you get five Dune sequels — each individually a great
match and collectively a useless recommendation, because a reader who liked Dune
has probably already found them.

MMR greedily builds the result list, at each step preferring the candidate that
is relevant *and* unlike what has already been picked:

    next = argmax  [ λ·score(i) − (1−λ) · max_{j ∈ selected} cos(e_i, e_j) ]

λ = 0.7 keeps relevance dominant while still breaking up clusters. λ = 1.0
degenerates to plain score ordering.

Operates on the embeddings already fetched by retrieval, so it costs no extra
round trip — for a 50-candidate pool that is a 50x50 similarity matrix, a few
hundred microseconds in numpy.
"""

import math
from collections.abc import Iterable, Sequence
from typing import Optional, cast

import numpy as np
import numpy.typing as npt

DEFAULT_LAMBDA = 0.70
FloatArray = npt.NDArray[np.float64]


def _normalize_rows(matrix: FloatArray) -> FloatArray:
    """L2-normalize so a dot product is a cosine.

    Zero-norm rows are left alone rather than producing NaN — a missing embedding
    should make an item maximally *dissimilar* to everything (so diversity never
    blocks it), not poison the whole matrix.
    """
    norms = cast(FloatArray, np.linalg.norm(matrix, axis=1, keepdims=True))
    norms[norms == 0] = 1.0
    return cast(FloatArray, matrix / norms)


def mmr_select(
    embeddings: Sequence[Sequence[float]],
    scores: Sequence[float],
    k: int,
    lambda_: float = DEFAULT_LAMBDA,
) -> list[int]:
    """Select up to `k` indices by MMR, best first.

    Returns indices into the input sequences so the caller can keep whatever row
    representation it already has.
    """
    n = len(scores)
    if n == 0 or k <= 0:
        return []
    k = min(int(k), n)
    lambda_ = max(0.0, min(1.0, float(lambda_)))

    relevance = cast(FloatArray, np.asarray(scores, dtype=np.float64))

    if lambda_ >= 1.0 or not embeddings:
        # No diversity pressure — plain score order. Short-circuited so the
        # degenerate case does no matrix work.
        return sorted(range(n), key=lambda index: -scores[index])[:k]

    matrix = _normalize_rows(cast(FloatArray, np.asarray(embeddings, dtype=np.float64)))
    similarity = cast(FloatArray, matrix @ matrix.T)

    selected: list[int] = [int(np.argmax(relevance))]
    # Running max similarity to the selected set, updated incrementally: an O(n)
    # update per pick instead of rescanning the selected set each time.
    first_similarity_row = cast(FloatArray, similarity[selected[0]])
    max_sim = first_similarity_row.copy()

    while len(selected) < k:
        objective = cast(FloatArray, lambda_ * relevance - (1.0 - lambda_) * max_sim)
        objective[selected] = -np.inf
        nxt = int(np.argmax(objective))
        if not math.isfinite(cast(float, objective[nxt])):
            break
        selected.append(nxt)
        next_similarity_row = cast(FloatArray, similarity[nxt])
        max_sim = cast(FloatArray, np.maximum(max_sim, next_similarity_row))

    return selected


def diversify(
    records: Sequence[dict],
    k: int,
    lambda_: float = DEFAULT_LAMBDA,
    embedding_field: str = "embedding",
    score_field: str = "score",
) -> list[dict]:
    """Reorder and trim scored records by MMR.

    Records missing an embedding are given a zero vector, which makes them
    maximally dissimilar to everything and so never suppressed for redundancy.
    That is the safe direction: an item we cannot compare should not be dropped
    for looking similar to something.
    """
    if not records:
        return []

    dimension = 0
    for record in records:
        vector = record.get(embedding_field)
        if vector is not None:
            dimension = len(vector)
            break

    embeddings = [
        list(record.get(embedding_field) or [0.0] * dimension) for record in records
    ]
    scores = [float(record.get(score_field) or 0.0) for record in records]

    if dimension == 0:
        # Nothing to compare on; fall back to score order rather than pretending
        # to diversify.
        ordered = sorted(range(len(records)), key=lambda i: -scores[i])[
            : max(0, int(k))
        ]
        return [records[i] for i in ordered]

    return [records[i] for i in mmr_select(embeddings, scores, k, lambda_)]


def intra_list_diversity(
    embeddings: Sequence[Sequence[float]],
) -> Optional[float]:
    """1 − mean pairwise cosine over a result list, in [0, 1].

    The online metric that shows MMR is doing something: a shelf of sequels scores
    near 0, a genuinely varied shelf nearer 1. None for fewer than two items,
    where diversity is undefined rather than perfect.
    """
    if len(embeddings) < 2:
        return None
    matrix = _normalize_rows(cast(FloatArray, np.asarray(embeddings, dtype=np.float64)))
    similarity = cast(FloatArray, matrix @ matrix.T)
    n = len(similarity)
    # Upper triangle only — the diagonal is self-similarity and the matrix is
    # symmetric, so including either would bias the mean upward.
    upper = similarity[np.triu_indices(n, k=1)]
    if upper.size == 0:
        return None
    mean_similarity = cast(float, upper.mean())
    return 1.0 - max(-1.0, min(1.0, mean_similarity))


__all__: Iterable[str] = [
    "DEFAULT_LAMBDA",
    "diversify",
    "intra_list_diversity",
    "mmr_select",
]
