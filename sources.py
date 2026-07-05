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
from typing import TYPE_CHECKING, Iterator, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    from storage.base_store import BaseVectorStore


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


class StorageBackedSource:
    """EmbeddingSource adapter over a loaded BaseVectorStore.

    Use this instead of NumpyFileSource once the storage layer has been built
    (``python scripts/build_storage.py``). Supports both Float32Store and
    SQ8Store; SQ8Store dequantizes on the fly so callers always get float32.

    Usage::

        from storage import Float32Store   # or SQ8Store
        from sources import StorageBackedSource

        store = Float32Store()
        store.load()
        source = StorageBackedSource(store)
        index.build(source)
    """

    def __init__(self, store: "BaseVectorStore"):
        if store.get_memory_footprint() == 0.0:
            raise RuntimeError(
                "Store appears empty — call store.load() before wrapping it."
            )
        self._store = store
        # Probe shape via a single-vector fetch so we don't depend on internals.
        sample = store.get_vectors(np.array([0], dtype=np.int64))
        self._dim = int(sample.shape[1])
        # Derive n_vectors from whichever attribute the store exposes.
        if hasattr(store, "data") and store.data is not None:
            self._n = int(store.data.shape[0])
        elif hasattr(store, "quantized_data") and store.quantized_data is not None:
            self._n = int(store.quantized_data.shape[0])
        else:
            raise RuntimeError("Cannot determine n_vectors from store.")
        self._ids = np.arange(self._n, dtype=np.int64)

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
            indices = self._ids[start:end]
            yield indices, self._store.get_vectors(indices)

    def get_by_ids(self, ids: np.ndarray) -> np.ndarray:
        return self._store.get_vectors(ids)
