"""Abstract search engine."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from .types import Hit, SearchResult


class BaseSearchEngine(ABC):
    """Abstract cosine-similarity search over a fixed corpus."""

    @abstractmethod
    def search(self, query: np.ndarray, top_k: int = 5) -> SearchResult:
        """Return top-``top_k`` documents ranked by cosine similarity.

        Parameters
        ----------
        query
            shape is ``(dim,)``
        top_k
            Number of hits to return. Result list may be shorter if fewer
            candidates are available.
        """

    @staticmethod
    def _build_result(ids: np.ndarray, scores: np.ndarray) -> SearchResult:
        """Package ``(ids, scores)`` arrays into the SearchResult contract."""
        hits: list[Hit] = [
            {"id": int(i), "score": float(s)} for i, s in zip(ids, scores)
        ]
        max_score = float(scores[0]) if len(scores) > 0 else 0.0
        return {"max_score": max_score, "results": hits}
