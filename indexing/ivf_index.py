"""Wrapper around FAISS IVFFlat index for production use."""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import faiss
import numpy as np


@dataclass
class IVFFlatConfig:
    """Configuration for the IVFFlat index.

    Attributes
    ----------
    dim
        Embedding dimensionality.
    n_list
        Number of Voronoi cells (centroids).
    n_probe
        How many cells to visit at query time. Runtime knob. 
    metric
        ``"cosine"`` (default) = IP on L2-normalized inputs.
        ``"ip"`` = raw inner product. ``"l2"`` = Euclidean.
    normalize
        Apply L2-normalization before FAISS.
    """

    dim: int
    n_list: int = 512
    nprobe: int = 16
    metric: Literal["cosine", "l2", "ip"] = "cosine"
    normalize: bool = True


class IVFFlatIndex:
    """Production IVFFlat index: inverted file with exact flat distances."""

    def __init__(self, cfg: IVFFlatConfig):
        self.cfg = cfg
        self._faiss_metric = self._resolve_metric()
        self._build_index()
        self._is_trained = False
        self._n_added = 0

    def _resolve_metric(self) -> int:
        return {
            "l2": faiss.METRIC_L2,
            "cosine": faiss.METRIC_INNER_PRODUCT,
            "ip": faiss.METRIC_INNER_PRODUCT,
        }[self.cfg.metric]

    def _build_index(self) -> None:
        quantizer = faiss.IndexFlatIP(self.cfg.dim) if self._faiss_metric == faiss.METRIC_INNER_PRODUCT \
            else faiss.IndexFlatL2(self.cfg.dim)
        self._index = faiss.IndexIVFFlat(
            quantizer, self.cfg.dim, self.cfg.n_list, self._faiss_metric
        )
        self._index.nprobe = int(self.cfg.nprobe)

    def _should_normalize(self) -> bool:
        return self.cfg.normalize and self.cfg.metric in ("cosine", "ip")

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self._should_normalize():
            norms = np.linalg.norm(x, axis=1, keepdims=True)
            x = x / (norms + 1e-12)
        return x

    def train(self, corpus: np.ndarray) -> None:
        """Run K-means on corpus to compute IVF centroids."""
        corpus = self._prepare(corpus)
        self._index.train(corpus)
        self._is_trained = True

    def add(self, corpus: np.ndarray) -> None:
        """Assign vectors to their nearest centroid and store."""
        if not self._is_trained:
            raise RuntimeError("Call train() before add().")
        corpus = self._prepare(corpus)
        self._index.add(corpus)
        self._n_added += len(corpus)

    def train_add(self, corpus: np.ndarray) -> None:
        """Convenience: train and add in one call."""
        self.train(corpus)
        self.add(corpus)

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(distances, labels)`` top-k for each query.

        Parameters
        ----------
        n_probe
            Overrides the config value at runtime
        """
        queries = self._prepare(queries)
        if nprobe is not None:
            self._index.nprobe = int(nprobe)
        return self._index.search(queries, k)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        cfg_path = path.with_suffix(path.suffix + ".cfg.pkl")
        with open(cfg_path, "wb") as f:
            pickle.dump(self.cfg, f)

    @classmethod
    def load(cls, path: str | Path) -> "IVFFlatIndex":
        path = Path(path)
        cfg_path = path.with_suffix(path.suffix + ".cfg.pkl")
        with open(cfg_path, "rb") as f:
            cfg = pickle.load(f)
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj._faiss_metric = obj._resolve_metric()
        obj._build_index()
        obj._index = faiss.read_index(str(path))
        obj._index.nprobe = int(cfg.nprobe)
        obj._is_trained = True
        obj._n_added = obj._index.ntotal
        return obj

    @property
    def n_vectors(self) -> int:
        return int(self._n_added)

    @property
    def bytes_per_vector(self) -> int:
        """Uncompressed footprint per vector (fp32) — IVFFlat stores full vectors."""
        return self.cfg.dim * 4

    def __repr__(self) -> str:
        return (
            f"IVFFlatIndex(dim={self.cfg.dim}, n_list={self.cfg.n_list}, "
            f"n_probe={self.cfg.nprobe}, metric='{self.cfg.metric}', "
            f"normalize={self._should_normalize()}, n={self.n_vectors}, "
            f"bytes/vec={self.bytes_per_vector})"
        )
