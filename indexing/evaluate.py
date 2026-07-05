"""Recall@k evaluation against brute-force ground truth.

Two entry points:

    compute_ground_truth(corpus, queries, k, metric)
        Runs FAISS ``IndexFlat*`` as reference the approximate
        indices are measured against.

    evaluate_recall(index, queries, ground_truth, k)
        Runs the index's search and returns recall@k + latency stats.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

import faiss
import numpy as np


class SearchableIndex(Protocol):
    """Minimal protocol satisfied by any FAISS-like index."""

    def search(
        self, queries: np.ndarray, k: int
    ) -> tuple[np.ndarray, np.ndarray]: ...


@dataclass
class EvalResult:
    """Result of a single evaluate_recall call."""

    n_queries: int
    k: int
    recall_at_k: float
    latency_ms_mean: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float

    def as_dict(self) -> dict:
        return {
            "n_queries": self.n_queries,
            "k": self.k,
            "recall_at_k": round(self.recall_at_k, 4),
            "latency_ms_mean": round(self.latency_ms_mean, 4),
            "latency_ms_p50": round(self.latency_ms_p50, 4),
            "latency_ms_p95": round(self.latency_ms_p95, 4),
            "latency_ms_p99": round(self.latency_ms_p99, 4),
        }


def compute_ground_truth(
    corpus: np.ndarray,
    queries: np.ndarray,
    k: int,
    metric: str = "cosine",
) -> np.ndarray:
    """Brute-force top-k for the given metric.

    Returns an ``(n_queries, k)`` int64 array of corpus positions,
    sorted best-to-worst per row.

    Metrics:
        ``cosine`` — normalizes both sides, uses ``IndexFlatIP``
        ``ip``     — raw inner product, ``IndexFlatIP``
        ``l2``     — Euclidean, ``IndexFlatL2``
    """
    corpus = np.ascontiguousarray(corpus, dtype=np.float32)
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    d = corpus.shape[1]

    if metric == "cosine":
        c = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-12)
        q = queries / (np.linalg.norm(queries, axis=1, keepdims=True) + 1e-12)
        idx = faiss.IndexFlatIP(d)
        idx.add(c.astype(np.float32))
        _, gt = idx.search(q.astype(np.float32), k)
    elif metric == "ip":
        idx = faiss.IndexFlatIP(d)
        idx.add(corpus)
        _, gt = idx.search(queries, k)
    elif metric == "l2":
        idx = faiss.IndexFlatL2(d)
        idx.add(corpus)
        _, gt = idx.search(queries, k)
    else:
        raise ValueError(f"Unknown metric: {metric!r}")
    return gt


def evaluate_recall(
    index: SearchableIndex,
    queries: np.ndarray,
    ground_truth: np.ndarray,
    k: int,
    n_warmup: int = 5,
    search_kwargs: Optional[dict] = None,
) -> EvalResult:
    """Measure recall@k and per-query latency percentiles.

    ``search_kwargs`` is forwarded to ``index.search(queries, k, **kwargs)``
    on every call — use it to pin runtime parameters like ``nprobe`` or
    ``ef_search`` for the specific config being tested.
    """
    search_kwargs = search_kwargs or {}
    queries = np.ascontiguousarray(queries, dtype=np.float32)

    # Warmup — mostly matters for HNSW graph traversal caches.
    for _ in range(n_warmup):
        _ = index.search(queries[:1], k, **search_kwargs)

    # Per-query timing (individual calls give honest latency, not
    # batch-throughput divided by N).
    per_query_ms: list[float] = []
    all_labels: list[np.ndarray] = []
    for q in queries:
        t = time.perf_counter()
        _, labels = index.search(q.reshape(1, -1), k, **search_kwargs)
        per_query_ms.append((time.perf_counter() - t) * 1000)
        all_labels.append(labels[0])

    labels_arr = np.stack(all_labels)
    hits = 0
    for i in range(len(labels_arr)):
        hits += len(set(labels_arr[i].tolist()) & set(ground_truth[i].tolist()))
    recall = hits / (len(labels_arr) * k)

    latencies = np.asarray(per_query_ms)
    return EvalResult(
        n_queries=len(queries),
        k=k,
        recall_at_k=recall,
        latency_ms_mean=float(latencies.mean()),
        latency_ms_p50=float(np.percentile(latencies, 50)),
        latency_ms_p95=float(np.percentile(latencies, 95)),
        latency_ms_p99=float(np.percentile(latencies, 99)),
    )


class Evaluator:
    """Reusable eval harness — computes GT once, evaluates many indices.

    Typical usage in a sweep::

        ev = Evaluator(corpus, queries, k=10, metric="cosine")
        for nprobe in [4, 16, 64]:
            r = ev.eval(ivfopq, search_kwargs={"nprobe": nprobe})
            print(nprobe, r.recall_at_k, r.latency_ms_mean)
    """

    def __init__(
        self,
        corpus: np.ndarray,
        queries: np.ndarray,
        k: int = 10,
        metric: str = "cosine",
    ):
        self.corpus = np.ascontiguousarray(corpus, dtype=np.float32)
        self.queries = np.ascontiguousarray(queries, dtype=np.float32)
        self.k = int(k)
        self.metric = metric
        self.ground_truth = compute_ground_truth(
            self.corpus, self.queries, self.k, metric
        )

    def eval(
        self,
        index: SearchableIndex,
        k: Optional[int] = None,
        search_kwargs: Optional[dict] = None,
        n_warmup: int = 5,
    ) -> EvalResult:
        k = int(k) if k is not None else self.k
        gt_k = self.ground_truth[:, :k]
        return evaluate_recall(
            index, self.queries, gt_k, k,
            n_warmup=n_warmup, search_kwargs=search_kwargs,
        )

    def sweep(
        self,
        index: SearchableIndex,
        param_name: str,
        param_values: list,
        k: Optional[int] = None,
    ) -> list[dict]:
        """Sweep one search-time parameter, return list of result dicts."""
        results = []
        for v in param_values:
            r = self.eval(index, k=k, search_kwargs={param_name: v})
            row = {param_name: v, **r.as_dict()}
            results.append(row)
        return results
