"""Cluster-quality diagnostics for IVF indices.

These functions answer the Level-1 research question: *how balanced are
the clusters produced by a given kmeans configuration?* Imbalance is
the primary driver of tail latency in IVF — if 5% of the queries land
in 50% of the corpus, your p99 collapses.

All functions operate on an ``IVFIndex`` that has been built. They are
pure: no side effects, no I/O.
"""

from __future__ import annotations

import numpy as np

from indexing.index import IVFIndex
from indexing.sources import EmbeddingSource


def cluster_sizes(index: IVFIndex) -> np.ndarray:
    """Return an ``(n_clusters,)`` int array of vectors-per-cluster.

    Includes empty clusters as zeros. Position ``i`` in the returned
    array corresponds to centroid ``i``.
    """
    if index.assignments is None:
        raise RuntimeError("Index has not been built.")
    sizes = np.zeros(index.cfg.n_clusters, dtype=np.int64)
    unique, counts = np.unique(index.assignments, return_counts=True)
    sizes[unique] = counts
    return sizes


def imbalance_metrics(index: IVFIndex) -> dict[str, float]:
    """Summarize the cluster-size distribution.

    Metrics:
        - ``n_empty``       : count of empty clusters
        - ``min`` / ``max`` : smallest / largest cluster size
        - ``mean`` / ``std`` : moments
        - ``cv``            : coefficient of variation (std / mean) — scale-free
                              imbalance score; 0 = perfectly balanced
        - ``max_over_mean`` : worst-case latency multiplier vs ideal
        - ``gini``          : Gini coefficient of the size distribution; 0 =
                              equal, 1 = all vectors in one cluster
        - ``entropy_norm``  : Shannon entropy of size distribution, normalized
                              to ``[0, 1]``. 1 = uniform, 0 = concentrated.
        - ``p50`` / ``p95`` / ``p99`` : percentile cluster sizes
    """
    sizes = cluster_sizes(index)
    nonzero = sizes[sizes > 0]
    n = int(sizes.sum())

    metrics: dict[str, float] = {
        "n_clusters": float(len(sizes)),
        "n_empty": float((sizes == 0).sum()),
        "min": float(sizes.min()),
        "max": float(sizes.max()),
        "mean": float(sizes.mean()),
        "std": float(sizes.std()),
        "cv": float(sizes.std() / (sizes.mean() + 1e-12)),
        "max_over_mean": float(sizes.max() / (sizes.mean() + 1e-12)),
        "gini": _gini(sizes),
        "entropy_norm": _entropy_norm(nonzero, k=len(sizes)),
        "p50": float(np.percentile(sizes, 50)),
        "p95": float(np.percentile(sizes, 95)),
        "p99": float(np.percentile(sizes, 99)),
        "total_vectors": float(n),
    }
    return metrics


def _gini(sizes: np.ndarray) -> float:
    """Gini coefficient of a non-negative array.

    Computed via the relative mean absolute difference, which works for
    any size distribution including zeros. Returns 0 for uniform sizes,
    approaches 1 as concentration grows.
    """
    if sizes.sum() == 0:
        return 0.0
    s = np.sort(sizes.astype(np.float64))
    n = len(s)
    cum = np.cumsum(s)
    # Standard Gini formula:  G = (2 * sum(i * s_i) / (n * sum(s))) - (n+1)/n
    idx = np.arange(1, n + 1)
    return float((2 * np.sum(idx * s) / (n * cum[-1])) - (n + 1) / n)


def compute_inertia(
    index: IVFIndex, source: EmbeddingSource, batch_size: int = 8192
) -> float:
    """Cosine-based inertia: ``sum(1 - cos(x, c_x))`` over all vectors.

    Works as a unified quality score across clusterers — both standard
    and spherical kmeans operate on L2-normalized inputs in this module,
    so cosine distance is the right thing to minimize. Lower is better.

    Implementation streams the source once, projects each batch through
    the index's MRL pipeline, and accumulates the per-vector distance to
    the assigned centroid.
    """
    if index.centroids is None or index.assignments is None:
        raise RuntimeError("Index has not been built.")

    total = 0.0
    n_seen = 0
    for _ids, vecs in source.iter_batches(batch_size):
        vecs = index._project(vecs)
        end = n_seen + len(vecs)
        assigned = index.assignments[n_seen:end]
        # Per-vector cosine sim between x and its assigned centroid.
        sims = np.einsum("ij,ij->i", vecs, index.centroids[assigned])
        total += float((1.0 - sims).sum())
        n_seen = end
    return total


def _entropy_norm(nonzero_sizes: np.ndarray, k: int) -> float:
    """Shannon entropy of the size distribution, normalized to ``[0, 1]``.

    Normalization divides by ``log(k)``, the entropy of a uniform
    distribution over ``k`` clusters. 1.0 = perfectly uniform, 0.0 = all
    mass on one cluster.
    """
    if len(nonzero_sizes) <= 1 or k <= 1:
        return 0.0
    p = nonzero_sizes / nonzero_sizes.sum()
    h = -float(np.sum(p * np.log(p)))
    return h / np.log(k)
