"""IVF (Inverted File) index — the main artifact this module produces.

Pipeline:
    EmbeddingSource  →  IVFIndex.build()  →  centroids + inverted lists
    query (768-dim)  →  IVFIndex.get_candidates()  →  list of candidate ids

The index owns the *structure* (centroids and which vectors live where).
It does NOT own:
    - the final cosine-similarity scoring (that's Search / Person 5)
    - vector storage in production (that's Storage / Person 3 — we use a
      lightweight backref by external id)

MRL: vectors are truncated to ``cfg.truncate_dim`` before clustering and
search. The source still provides full-dim vectors; truncation happens
inside this module.
"""

from __future__ import annotations

import json
import pickle
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from indexing.clustering import Clusterer, KMeansConfig, SklearnMiniBatchKMeans
from indexing.sources import EmbeddingSource


@dataclass
class IVFConfig:
    """Index-level configuration.

    Attributes
    ----------
    n_clusters
        Number of IVF cells. Rule of thumb: ``sqrt(N)``. For 500k that's
        ~707; 512 is a power-of-two approximation.
    truncate_dim
        MRL slice size. ``None`` keeps full dimensionality. EmbeddingGemma
        guarantees {128, 256, 512, 768} as supported MRL steps.
    normalize
        L2-normalize vectors after truncation. Required for cosine
        similarity to behave correctly on the truncated subspace.
    sample_size_for_training
        How many vectors to sample from the source for kmeans training.
        ``None`` = use everything. Sampling is fine for kmeans because
        centroids are statistical — you don't need every single point.
    """

    n_clusters: int = 512
    truncate_dim: Optional[int] = 256
    normalize: bool = True
    sample_size_for_training: Optional[int] = None
    seed: int = 42


@dataclass
class BuildStats:
    """Diagnostics from the most recent ``build`` call. Empty until built."""

    n_vectors: int = 0
    source_dim: int = 0
    index_dim: int = 0
    n_clusters: int = 0
    build_seconds: float = 0.0
    training_sample_size: int = 0
    clusterer_class: str = ""
    timings: dict = field(default_factory=dict)


class IVFIndex:
    """Inverted-file index over dense embeddings.

    Lifecycle:
        1. ``build(source)`` — train centroids, assign all source vectors
        2. ``get_candidates(query, nprobe)`` — query-time hot path
        3. ``add(vec_id, vec)`` — incremental add without retraining
        4. ``save(path)`` / ``load(path)`` — persist
    """

    def __init__(self, cfg: IVFConfig, clusterer: Optional[Clusterer] = None):
        self.cfg = cfg
        self.clusterer = clusterer or SklearnMiniBatchKMeans(
            KMeansConfig(n_clusters=cfg.n_clusters, seed=cfg.seed)
        )

        # State populated by build / load.
        self.centroids: Optional[np.ndarray] = None  # (k, d_idx) float32
        self.assignments: Optional[np.ndarray] = None  # (N,) int32 — cluster id
        self.vec_ids: Optional[np.ndarray] = None  # (N,) int64 — external ids
        self.inverted_lists: dict[int, list[int]] = {}  # cid -> [positions, ...]
        self.stats: BuildStats = BuildStats()

    # ------------------------------------------------------------------
    # Building
    # ------------------------------------------------------------------

    def build(self, source: EmbeddingSource) -> None:
        """Train centroids and assign all vectors from ``source``.

        Steps:
            1. stream vectors out of source, truncate, normalize
            2. train clusterer on (a sample of) those vectors
            3. assign every vector to its nearest centroid
            4. build the inverted lists dict
        """
        t0 = time.perf_counter()
        timings: dict[str, float] = {}

        # ---- 1. Materialize and preprocess all vectors ----
        t = time.perf_counter()
        ids, vecs = self._stream_all(source)
        timings["stream"] = time.perf_counter() - t

        d_idx = vecs.shape[1]

        # ---- 2. Train clusterer ----
        t = time.perf_counter()
        if (
            self.cfg.sample_size_for_training is not None
            and self.cfg.sample_size_for_training < len(vecs)
        ):
            rng = np.random.default_rng(self.cfg.seed)
            sample_idx = rng.choice(
                len(vecs), size=self.cfg.sample_size_for_training, replace=False
            )
            train_X = vecs[sample_idx]
        else:
            train_X = vecs

        centroids, _ = self.clusterer.fit(train_X)
        timings["train"] = time.perf_counter() - t

        if self.cfg.normalize:
            centroids = self._l2_normalize(centroids)
        self.centroids = centroids.astype(np.float32)

        # ---- 3. Assign all vectors to centroids ----
        t = time.perf_counter()
        assignments = self._assign_in_batches(vecs)
        timings["assign"] = time.perf_counter() - t

        # ---- 4. Build inverted lists ----
        t = time.perf_counter()
        self.assignments = assignments
        self.vec_ids = ids
        self.inverted_lists = self._build_inverted_lists(assignments, self.cfg.n_clusters)
        timings["invert"] = time.perf_counter() - t

        self.stats = BuildStats(
            n_vectors=len(vecs),
            source_dim=source.dim,
            index_dim=d_idx,
            n_clusters=self.cfg.n_clusters,
            build_seconds=time.perf_counter() - t0,
            training_sample_size=len(train_X),
            clusterer_class=type(self.clusterer).__name__,
            timings=timings,
        )

    def _stream_all(
        self, source: EmbeddingSource, batch_size: int = 8192
    ) -> tuple[np.ndarray, np.ndarray]:
        """Pull all vectors out, truncate to MRL slice, optionally normalize."""
        all_ids: list[np.ndarray] = []
        all_vecs: list[np.ndarray] = []
        for ids, vecs in source.iter_batches(batch_size):
            vecs = self._project(vecs)
            all_ids.append(ids)
            all_vecs.append(vecs)
        return np.concatenate(all_ids), np.concatenate(all_vecs)

    def _project(self, vecs: np.ndarray) -> np.ndarray:
        """Apply MRL truncation and (optional) L2 normalization."""
        if self.cfg.truncate_dim is not None:
            vecs = vecs[:, : self.cfg.truncate_dim]
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if self.cfg.normalize:
            vecs = self._l2_normalize(vecs)
        return vecs

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
        return x / norms

    # ------------------------------------------------------------------
    # Assignment
    # ------------------------------------------------------------------

    def _assign_in_batches(
        self, vecs: np.ndarray, batch_size: int = 16_384
    ) -> np.ndarray:
        """Assign each vector to its nearest centroid in (n, k) chunks.

        We chunk to avoid allocating an ``(N, k)`` similarity matrix all
        at once — for N=500k and k=512 that's a 1 GB float32 block.
        """
        assert self.centroids is not None
        n = len(vecs)
        out = np.empty(n, dtype=np.int32)
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            sims = vecs[start:end] @ self.centroids.T  # cosine sim on unit vectors
            out[start:end] = sims.argmax(axis=1)
        return out

    @staticmethod
    def _build_inverted_lists(
        assignments: np.ndarray, n_clusters: int
    ) -> dict[int, list[int]]:
        lists: dict[int, list[int]] = {c: [] for c in range(n_clusters)}
        for pos, cid in enumerate(assignments):
            lists[int(cid)].append(pos)
        return lists

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_candidates(self, query: np.ndarray, nprobe: int = 8) -> np.ndarray:
        """Return external vec_ids of all candidates for the query.

        The query is projected (truncated + normalized) using the same
        rules as build, so callers can pass the raw source-dim vector.

        Returns an unsorted ``np.ndarray`` of int64 ids. Sorting and
        final cosine ranking are the Search component's job.
        """
        if self.centroids is None or self.vec_ids is None:
            raise RuntimeError("Index has not been built or loaded.")

        q = self._project(query.reshape(1, -1))[0]  # (d_idx,)
        sims = self.centroids @ q  # (k,)
        nprobe = min(nprobe, len(sims))
        top_cells = np.argpartition(-sims, nprobe - 1)[:nprobe]

        # Concatenate inverted lists from top cells, map to external ids.
        positions: list[int] = []
        for cid in top_cells:
            positions.extend(self.inverted_lists.get(int(cid), ()))
        if not positions:
            return np.empty(0, dtype=np.int64)
        return self.vec_ids[np.asarray(positions, dtype=np.int64)]

    # ------------------------------------------------------------------
    # Incremental add
    # ------------------------------------------------------------------

    def add(self, vec_id: int, vec: np.ndarray) -> None:
        """Add a single vector without retraining centroids.

        Centroid drift over time: each ``add`` makes the index slightly
        more stale relative to true cluster structure. Plan for periodic
        full retraining when the corpus shifts substantially.
        """
        if self.centroids is None or self.vec_ids is None:
            raise RuntimeError("Index has not been built or loaded.")

        q = self._project(vec.reshape(1, -1))[0]
        cid = int((self.centroids @ q).argmax())

        position = len(self.vec_ids)
        self.vec_ids = np.append(self.vec_ids, np.int64(vec_id))
        assert self.assignments is not None
        self.assignments = np.append(self.assignments, np.int32(cid))
        self.inverted_lists.setdefault(cid, []).append(position)
        self.stats.n_vectors += 1

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Persist as a directory with separate artifacts.

        Layout::

            path/
                centroids.npy
                assignments.npy
                vec_ids.npy
                inverted_lists.pkl
                config.json     (cfg + stats, for provenance)
        """
        if self.centroids is None:
            raise RuntimeError("Nothing to save — index has not been built.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "centroids.npy", self.centroids)
        np.save(path / "assignments.npy", self.assignments)
        np.save(path / "vec_ids.npy", self.vec_ids)
        with open(path / "inverted_lists.pkl", "wb") as f:
            pickle.dump(self.inverted_lists, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(path / "config.json", "w") as f:
            json.dump(
                {"config": asdict(self.cfg), "stats": asdict(self.stats)},
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str | Path) -> "IVFIndex":
        path = Path(path)
        with open(path / "config.json") as f:
            payload = json.load(f)

        cfg = IVFConfig(**payload["config"])
        idx = cls(cfg)
        idx.centroids = np.load(path / "centroids.npy")
        idx.assignments = np.load(path / "assignments.npy")
        idx.vec_ids = np.load(path / "vec_ids.npy")
        with open(path / "inverted_lists.pkl", "rb") as f:
            idx.inverted_lists = pickle.load(f)

        stats_dict = payload.get("stats", {})
        if stats_dict:
            idx.stats = BuildStats(**stats_dict)
        return idx
