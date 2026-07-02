"""Clustering algorithms — the heart of IVF training.

Strategy pattern: ``IVFIndex`` doesn't know which kmeans is running. You
inject a ``Clusterer``, and the index calls ``fit(X) -> (centroids, labels)``.

For the Level-1 experiments (random vs k-means++ initialization, full vs
mini-batch) you swap the implementation, not the index code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np


@dataclass
class KMeansConfig:
    """Hyperparameters shared by all kmeans variants."""

    n_clusters: int = 512
    init: Literal["random", "k-means++"] = "k-means++"
    max_iter: int = 100
    n_init: int = 1
    tol: float = 1e-4
    seed: int = 42


class Clusterer(Protocol):
    """Strategy interface for clustering."""

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Cluster ``X`` shape ``(N, d)``.

        Returns ``(centroids, labels)`` where ``centroids`` is ``(k, d)``
        float32 and ``labels`` is ``(N,)`` int32.
        """
        ...


# ---------------------------------------------------------------------------
# Sklearn-backed implementations
# ---------------------------------------------------------------------------


class SklearnKMeans:
    """Full Lloyd's k-means via ``sklearn.cluster.KMeans``.

    Accurate. Slow on 500k. Use as the gold-standard reference for the
    init-comparison experiment.
    """

    def __init__(self, cfg: KMeansConfig):
        self.cfg = cfg

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.cluster import KMeans

        km = KMeans(
            n_clusters=self.cfg.n_clusters,
            init=self.cfg.init,
            n_init=self.cfg.n_init,
            max_iter=self.cfg.max_iter,
            tol=self.cfg.tol,
            random_state=self.cfg.seed,
        )
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_.astype(np.float32)
        return centroids, labels.astype(np.int32)


class SklearnMiniBatchKMeans:
    """Mini-batch k-means — required for 500k+ vectors.

    Trades a bit of cluster quality for ~10x speedup. The recommended
    default for production-sized corpora.
    """

    def __init__(self, cfg: KMeansConfig, batch_size: int = 4096):
        self.cfg = cfg
        self.batch_size = batch_size

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from sklearn.cluster import MiniBatchKMeans

        km = MiniBatchKMeans(
            n_clusters=self.cfg.n_clusters,
            init=self.cfg.init,
            n_init=self.cfg.n_init,
            max_iter=self.cfg.max_iter,
            tol=self.cfg.tol,
            batch_size=self.batch_size,
            random_state=self.cfg.seed,
        )
        labels = km.fit_predict(X)
        centroids = km.cluster_centers_.astype(np.float32)
        return centroids, labels.astype(np.int32)



class SphericalKMeans:
    """K-means on L2-normalized vectors using cosine distance.

    Standard kmeans minimizes squared L2 distance, which is *almost* but
    not *exactly* right for cosine-similarity search. Spherical kmeans
    minimizes ``1 - cos(x, c)`` directly.

    Implementation:
        - centroids stay L2-normalized after every update
        - assignment = argmax of ``X @ C.T`` (since both are unit-norm)
        - update = mean of assigned vectors, then renormalize

    Use this if Person 2 confirms embeddings come pre-normalized — which
    they should, because EmbeddingGemma+MRL operates on the unit sphere.
    """

    def __init__(self, cfg: KMeansConfig):
        self.cfg = cfg

    def fit(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n, d = X.shape
        k = self.cfg.n_clusters
        rng = np.random.default_rng(self.cfg.seed)

        # Ensure X is on the sphere. Cheap if already normalized.
        norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        X = (X / norms).astype(np.float32)

        # Init: random sample or k-means++ on the sphere.
        if self.cfg.init == "random":
            idx = rng.choice(n, size=k, replace=False)
            centroids = X[idx].copy()
        elif self.cfg.init == "k-means++":
            centroids = self._kmeans_pp_init(X, k, rng)
        else:
            raise ValueError(f"Unknown init: {self.cfg.init}")

        labels = np.zeros(n, dtype=np.int32)
        prev_inertia = np.inf

        for _ in range(self.cfg.max_iter):
            # Assignment step. Cosine sim == dot product on unit vectors.
            sims = X @ centroids.T  # (n, k)
            labels = sims.argmax(axis=1).astype(np.int32)

            # Update step: mean of each cluster, then renormalize.
            new_centroids = np.zeros_like(centroids)
            counts = np.zeros(k, dtype=np.int64)
            np.add.at(new_centroids, labels, X)
            np.add.at(counts, labels, 1)

            empty = counts == 0
            if empty.any():
                # Reinit empty clusters to random points.
                reinit = rng.choice(n, size=int(empty.sum()), replace=False)
                new_centroids[empty] = X[reinit]
                counts[empty] = 1

            new_centroids /= counts[:, None]
            new_centroids /= (
                np.linalg.norm(new_centroids, axis=1, keepdims=True) + 1e-12
            )

            inertia = float((1.0 - sims.max(axis=1)).sum())
            shift = float(np.linalg.norm(new_centroids - centroids))
            centroids = new_centroids

            if abs(prev_inertia - inertia) < self.cfg.tol or shift < self.cfg.tol:
                break
            prev_inertia = inertia

        return centroids.astype(np.float32), labels

    @staticmethod
    def _kmeans_pp_init(X: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
        n = X.shape[0]
        first = int(rng.integers(0, n))
        centroids = [X[first]]
        # Squared cosine distance to the nearest chosen centroid.
        closest_sim = X @ centroids[0]
        for _ in range(1, k):
            dist_sq = (1.0 - closest_sim).clip(min=0) ** 2
            total = dist_sq.sum()
            if total <= 0:
                idx = int(rng.integers(0, n))
            else:
                probs = dist_sq / total
                idx = int(rng.choice(n, p=probs))
            centroids.append(X[idx])
            new_sim = X @ centroids[-1]
            closest_sim = np.maximum(closest_sim, new_sim)
        return np.stack(centroids).astype(np.float32)
