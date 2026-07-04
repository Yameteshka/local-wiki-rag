"""HNSW graph index — production recipe for high-recall vector search.

Standard settings for a 100k-1M corpus of L2-normalized text embeddings:

    M = 32                    graph degree (higher = better recall, more memory)
    ef_construction = 200     build quality (higher = slower build, better graph)
    ef_search = 64            runtime quality (higher = better recall, slower query)

Trade-off vs IVF-OPQ-PQ:
    - HNSW higher recall, higher memory
    - IVF-OPQ-PQ lower recall, ~30× less memory
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import faiss
import numpy as np


@dataclass
class HNSWConfig:
    """Configuration for the HNSW index.

    Attributes
    ----------
    dim
        Embedding dimensionality.
    M
        Graph degree (bidirectional edges per node). 16-64 is the useful
        range. Higher M = denser graph = better recall but more memory
        and slower build.
    ef_construction
        Search depth during index construction. Higher = better graph
        quality. 100-400 typical. Trained once, so pay upfront.
    ef_search
        Search depth at query time. Runtime knob. Higher = better recall
        but slower query. Sweep this at eval time.
    metric
        ``"cosine"`` (default) = IP on L2-normalized inputs.
        ``"ip"`` = raw inner product. ``"l2"`` = Euclidean.
    normalize
        Apply L2-normalization before FAISS. Required for ``metric="cosine"``.
    seed
        Reserved for future use — FAISS HNSW is deterministic given inputs.
    """

    dim: int
    M: int = 32
    ef_construction: int = 200
    ef_search: int = 64
    metric: Literal["cosine", "l2", "ip"] = "cosine"
    normalize: bool = True
    seed: int = 42


class HNSWIndex:
    """Production index: HNSW graph, no compression.
    """

    def __init__(self, cfg: HNSWConfig):
        self.cfg = cfg
        self._faiss_metric = self._resolve_metric()
        self._build_index()
        self._n_added = 0

    def _resolve_metric(self) -> int:
        return {
            "l2": faiss.METRIC_L2,
            "cosine": faiss.METRIC_INNER_PRODUCT,
            "ip": faiss.METRIC_INNER_PRODUCT,
        }[self.cfg.metric]

    def _build_index(self) -> None:
        self._index = faiss.IndexHNSWFlat(
            self.cfg.dim, self.cfg.M, self._faiss_metric
        )
        self._index.hnsw.efConstruction = int(self.cfg.ef_construction)
        self._index.hnsw.efSearch = int(self.cfg.ef_search)

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.cfg.normalize and self.cfg.metric == "cosine":
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        return x

    def train(self, corpus: np.ndarray) -> None:
        """No-op — HNSW builds its graph incrementally during add()."""
        # Kept for API symmetry with the other prod index.
        _ = corpus

    def add(self, corpus: np.ndarray) -> None:
        """Insert corpus into the graph. Graph is built incrementally."""
        corpus = self._prepare(corpus)
        self._index.add(corpus)
        self._n_added += len(corpus)

    def train_add(self, corpus: np.ndarray) -> None:
        """Convenience: HNSW has no separate train phase, so just add()."""
        self.add(corpus)

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        ef_search: Optional[int] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, labels) top-k for each query.

        ``ef_search`` (optional override) — higher = better recall,
        slower query. Typical sweep: {16, 32, 64, 128, 256}.
        """
        queries = self._prepare(queries)
        if ef_search is not None:
            self._index.hnsw.efSearch = int(ef_search)
        return self._index.search(queries, k)


    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        cfg_path = path.with_suffix(path.suffix + ".cfg.pkl")
        with open(cfg_path, "wb") as f:
            pickle.dump(self.cfg, f)

    @classmethod
    def load(cls, path: str | Path) -> "HNSWIndex":
        path = Path(path)
        cfg_path = path.with_suffix(path.suffix + ".cfg.pkl")
        with open(cfg_path, "rb") as f:
            cfg = pickle.load(f)
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj._faiss_metric = obj._resolve_metric()
        obj._index = faiss.read_index(str(path))
        obj._n_added = obj._index.ntotal
        return obj

    @property
    def n_vectors(self) -> int:
        return int(self._n_added)

    @property
    def bytes_per_vector(self) -> int:
        """Uncompressed footprint per vector (fp32)."""
        return self.cfg.dim * 4

    def __repr__(self) -> str:
        return (
            f"HNSWIndex(dim={self.cfg.dim}, M={self.cfg.M}, "
            f"efC={self.cfg.ef_construction}, efS={self.cfg.ef_search}, "
            f"metric='{self.cfg.metric}', n={self.n_vectors}, "
            f"bytes/vec={self.bytes_per_vector})"
        )
