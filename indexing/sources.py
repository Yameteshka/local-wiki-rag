"""Embedding sources — the upstream of the indexing pipeline.

The Indexer never talks to disk/database directly. It pulls from an
EmbeddingSource

Contract (Protocol):
    dim         : int     — embedding dimensionality
    n_vectors   : int     — total vector count
    iter_batches: yields (ids, vectors) in chunks
    get_by_ids  : random access by external id (used for rerank by Search)

External ids are arbitrary int identifiers chosen by the source (e.g. row
indices in a parquet file). The index stores them verbatim and returns
them as candidate ids.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class EmbeddingSource(Protocol):
    """Read-only view over a corpus of embeddings."""

    @property
    def dim(self) -> int: ...

    @property
    def n_vectors(self) -> int: ...

    def iter_batches(
        self, batch_size: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Yield ``(ids, vectors)``.

        ``ids`` shape ``(B,)`` int64. ``vectors`` shape ``(B, dim)`` float32.
        Last batch may be smaller than ``batch_size``.
        """
        ...

    def get_by_ids(self, ids: np.ndarray) -> np.ndarray:
        """Random-access fetch. ``ids`` shape ``(M,)``. Returns ``(M, dim)`` float32."""
        ...


class MockSource:
    """Gaussian-mixture on the unit hypersphere.

    Generates ``n`` vectors clustered around ``n_clusters_true`` hidden
    centroids — gives the kmeans something real to recover, unlike pure
    uniform noise which collapses cluster quality metrics.

    All vectors are L2-normalized so they live on the unit sphere, matching
    how real sentence embeddings are typically distributed.
    """

    def __init__(
        self,
        n: int = 500_000,
        dim: int = 768,
        n_clusters_true: int = 50,
        spread: float = 0.15,
        seed: int = 42,
    ):
        self._n = int(n)
        self._dim = int(dim)
        self._seed = int(seed)
        rng = np.random.default_rng(seed)

        # True hidden centroids on the unit sphere.
        true_centroids = rng.standard_normal((n_clusters_true, dim)).astype(np.float32)
        true_centroids /= np.linalg.norm(true_centroids, axis=1, keepdims=True) + 1e-12

        # Assign each vector to a hidden centroid (roughly uniform).
        labels = rng.integers(0, n_clusters_true, size=self._n)
        noise = rng.standard_normal((self._n, dim)).astype(np.float32) * spread
        vecs = true_centroids[labels] + noise
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12

        self._vectors = vecs
        self._ids = np.arange(self._n, dtype=np.int64)
        # Surfaced for tests / diagnostics. Don't rely on this in production code.
        self._true_labels = labels.astype(np.int32)

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def n_vectors(self) -> int:
        return self._n

    def iter_batches(
        self, batch_size: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        for start in range(0, self._n, batch_size):
            end = min(start + batch_size, self._n)
            yield self._ids[start:end], self._vectors[start:end]

    def get_by_ids(self, ids: np.ndarray) -> np.ndarray:
        # external id == position in MockSource (chosen for simplicity)
        return self._vectors[ids]


# ---------------------------------------------------------------------------
# NumpyFileSource: load from a .npy / .npz on disk
# ---------------------------------------------------------------------------


class NumpyFileSource:
    """Load embeddings from a ``.npy`` file.

    Expected layout: ``(N, dim)`` float32 array. Optional ``ids_path``
    points to a ``(N,)`` int64 array of external ids. If omitted, ids
    default to ``[0, 1, ..., N-1]``.

    Uses ``mmap_mode='r'`` so large files don't blow up RAM. Vectors are
    sliced lazily per batch.
    """

    def __init__(self, vectors_path: str | Path, ids_path: str | Path | None = None):
        vectors_path = Path(vectors_path)
        self._vectors = np.load(vectors_path, mmap_mode="r")
        if self._vectors.ndim != 2:
            raise ValueError(
                f"Expected 2D array in {vectors_path}, got shape {self._vectors.shape}"
            )

        if ids_path is not None:
            self._ids = np.load(Path(ids_path))
            if self._ids.shape[0] != self._vectors.shape[0]:
                raise ValueError(
                    f"ids length {self._ids.shape[0]} != vectors length "
                    f"{self._vectors.shape[0]}"
                )
        else:
            self._ids = np.arange(self._vectors.shape[0], dtype=np.int64)

    @property
    def dim(self) -> int:
        return int(self._vectors.shape[1])

    @property
    def n_vectors(self) -> int:
        return int(self._vectors.shape[0])

    def iter_batches(
        self, batch_size: int
    ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        n = self.n_vectors
        for start in range(0, n, batch_size):
            end = min(start + batch_size, n)
            # Copy to materialize from mmap and ensure float32.
            yield (
                np.asarray(self._ids[start:end]),
                np.asarray(self._vectors[start:end], dtype=np.float32),
            )

    def get_by_ids(self, ids: np.ndarray) -> np.ndarray:
        # NumpyFileSource ids are sequential — external id == row index.
        return np.asarray(self._vectors[ids], dtype=np.float32)
