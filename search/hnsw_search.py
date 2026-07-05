"""Iteration 3: HNSW graph-based ANN search. Uses the HNSWIndex from the indexing module."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from indexing.hnsw_index import HNSWIndex

from .base_search import BaseSearchEngine
from .types import SearchResult


@dataclass
class HNSWSearchConfig:
    """Runtime knob for the HNSW engine.

    Attributes
    ----------
    ef_search
        Search depth at query time. Higher = better recall, higher latency.
        Overrides the value baked into the underlying HNSWIndex if set.
    """

    ef_search: Optional[int] = None


class HNSWSearch(BaseSearchEngine):
    """Single-stage graph ANN. HNSW returns exact scores."""

    def __init__(
        self,
        hnsw_index: HNSWIndex,
        config: HNSWSearchConfig | None = None,
    ):
        self.hnsw = hnsw_index
        self.cfg = config or HNSWSearchConfig()

    def search(self, query: np.ndarray, top_k: int = 5) -> SearchResult:
        if query.ndim != 1:
            raise ValueError(f"Expected 1D query, got shape {query.shape}")

        # HNSWIndex handles normalization internally when cfg.metric == "cosine".
        q = query.reshape(1, -1).astype(np.float32, copy=False)
        scores, ids = self.hnsw.search(q, k=top_k, ef_search=self.cfg.ef_search)

        ids = ids[0]
        scores = scores[0]
        mask = ids >= 0
        return self._build_result(ids[mask], scores[mask])
