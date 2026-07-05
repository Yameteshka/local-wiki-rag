"""Benchmarking utilities for search engines."""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

import numpy as np
import psutil

from .base_search import BaseSearchEngine
from .brute_force import BruteForceSearch


@dataclass
class BenchmarkResult:
    name: str
    lat_mean_ms: float
    lat_p50_ms: float
    lat_p95_ms: float
    recall: float
    corpus_mb: float
    ram_mb: float
    latencies_ms: list[float] = field(default_factory=list)

    def as_row(self) -> str:
        return (
            f"{self.name:<22} "
            f"lat={self.lat_mean_ms:6.2f}ms  "
            f"p50={self.lat_p50_ms:6.2f}  p95={self.lat_p95_ms:6.2f}  "
            f"recall={self.recall:.3f}  "
            f"corpus={self.corpus_mb:6.1f}MB"
        )


def generate_random_queries(
    n_queries: int, dim: int, seed: int = 123
) -> list[np.ndarray]:
    """Draw ``n_queries`` unit-norm random query vectors of dim ``dim``."""
    rng = np.random.default_rng(seed)
    queries = []
    for _ in range(n_queries):
        q = rng.standard_normal(dim).astype(np.float32)
        q /= np.linalg.norm(q)
        queries.append(q)
    return queries


def compute_ground_truth(
    queries: list[np.ndarray], brute: BruteForceSearch, top_k: int
) -> list[set[int]]:
    """Exact top-K per query — the recall reference."""
    return [
        {hit["id"] for hit in brute.search(q, top_k=top_k)["results"]}
        for q in queries
    ]


def benchmark_engine(
    name: str,
    engine: BaseSearchEngine,
    queries: list[np.ndarray],
    ground_truth: list[set[int]],
    top_k: int,
    corpus_mb: float,
) -> BenchmarkResult:
    """Time each query and score recall against the ground truth."""
    latencies_ms: list[float] = []
    recalls: list[float] = []

    # warm up pulling; relatable only for cuda
    engine.search(queries[0], top_k=top_k)

    for q, gt in zip(queries, ground_truth):
        t0 = time.perf_counter()
        result = engine.search(q, top_k=top_k)
        latencies_ms.append((time.perf_counter() - t0) * 1000)

        pred_ids = {hit["id"] for hit in result["results"]}
        recalls.append(len(pred_ids & gt) / max(1, len(gt)))

    proc = psutil.Process(os.getpid())
    return BenchmarkResult(
        name=name,
        lat_mean_ms=float(np.mean(latencies_ms)),
        lat_p50_ms=float(np.percentile(latencies_ms, 50)),
        lat_p95_ms=float(np.percentile(latencies_ms, 95)),
        recall=float(np.mean(recalls)),
        corpus_mb=corpus_mb,
        ram_mb=proc.memory_info().rss / 1024**2,
        latencies_ms=latencies_ms,
    )


def print_summary(results: list[BenchmarkResult]) -> None:
    print("=" * 90)
    print("SEARCH VALIDATION SUMMARY")
    print("=" * 90)
    for r in results:
        print(r.as_row())
    print("=" * 90)
