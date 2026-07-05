"""Iteration 1: brute-force exact KNN via a single matmul."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .base_search import BaseSearchEngine
from .types import SearchResult


class BruteForceSearch(BaseSearchEngine):
    """Exact top-K over the full corpus. Ground truth for recall benchmarks."""

    def __init__(self, corpus: np.ndarray, device: str | None = None):
        """Build the engine from a raw (N, D) float32 corpus.

        """
        if corpus.ndim != 2:
            raise ValueError(f"Expected 2D corpus, got shape {corpus.shape}")

        self.device = device or (
            "cuda" if torch.cuda.is_available() else "cpu")

        normed = corpus.astype(np.float32, copy=True)
        normed /= np.linalg.norm(normed, axis=1, keepdims=True) + 1e-12
        self.corpus = torch.from_numpy(normed).to(self.device)

        self.n_vectors, self.dim = corpus.shape

    def search(self, query: np.ndarray, top_k: int = 5) -> SearchResult:
        if query.ndim != 1:
            raise ValueError(f"Expected 1D query, got shape {query.shape}")
        top_k = min(top_k, self.n_vectors)

        q = torch.from_numpy(query.astype(np.float32)).to(self.device)
        q = F.normalize(q, dim=0)

        scores = self.corpus @ q
        vals, idx = torch.topk(scores, k=top_k)

        if self.device == "cuda":
            torch.cuda.synchronize()

        return self._build_result(idx.cpu().numpy(), vals.cpu().numpy())
