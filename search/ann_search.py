"""Iteration 2: two-stage ANN search.

The rerank step compensates for the accuracy loss of PQ-compressed distance
estimates by pulling the original (or SQ8-dequantized) float32 vectors and
computing the true cosine similarity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from storage.base_store import BaseVectorStore

from .base_search import BaseSearchEngine
from .types import SearchResult


class IVFIndexLike(Protocol):
    """Minimal interface expected from an IVF-family index."""

    def search(
        self, queries: np.ndarray, k: int, nprobe: int = ...
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class ANNConfig:
    """Search-time knobs for the two-stage ANN pipeline.

    Attributes
    ----------
    n_probe
        Number of IVF cells to visit at query time.
    n_centroids
        Total number of IVF cells the index was trained with.
    candidate_multiplier
        Extra headroom on the candidate pool size. 2x is safe for skewed
        cluster distributions.
    """

    n_probe: int = 8
    n_centroids: int = 512
    candidate_multiplier: int = 2


class ANNSearch(BaseSearchEngine):
    """Two-stage ANN: IVF candidate generation + exact cosine rerank."""

    def __init__(
        self,
        ivf_index: IVFIndexLike,
        rerank_store: BaseVectorStore,
        n_vectors: int,
        config: ANNConfig | None = None,
    ):
        self.ivf = ivf_index
        self.store = rerank_store
        self.n_vectors = n_vectors
        self.cfg = config or ANNConfig()

    def search(self, query: np.ndarray, top_k: int = 5) -> SearchResult:
        if query.ndim != 1:
            raise ValueError(f"Expected 1D query, got shape {query.shape}")

        q = query.astype(np.float32, copy=True)
        q /= np.linalg.norm(q) + 1e-12

        # Stage 1: pull enough candidates to cover all vectors in visited cells.
        # todo: maybe we need to parameterize this logic
        avg_cell_size = max(1, self.n_vectors // self.cfg.n_centroids)
        n_candidates = min(
            self.n_vectors,
            self.cfg.n_probe * avg_cell_size * self.cfg.candidate_multiplier,
        )
        _, cand_ids = self.ivf.search(
            q.reshape(1, -1), k=n_candidates, nprobe=self.cfg.n_probe
        )
        cand_ids = cand_ids[0][cand_ids[0] >= 0]

        if len(cand_ids) == 0:
            return self._build_result(np.array([]), np.array([]))

        # Stage 2: exact cosine similarity via the rerank store.
        vecs = self.store.get_vectors(cand_ids)
        if vecs is None:
            raise RuntimeError("Rerank store returned None")

        vecs = vecs.astype(np.float32, copy=False)
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
        scores = vecs @ q

        k = min(top_k, len(scores))
        top_local = np.argpartition(-scores, k - 1)[:k]
        top_local = top_local[np.argsort(-scores[top_local])]

        return self._build_result(cand_ids[top_local], scores[top_local])
