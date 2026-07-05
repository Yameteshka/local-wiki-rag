""" This is wrapper around FAISS IVF-OPQ-PQ index for production use.

Pipeline:
    OPQ  — learned rotation, decorrelates dims for balanced PQ subvectors
    IVF  — spatial partitioning via k-means
    PQ   — compression to M × nbits bits per vector. Used oppositely to standardSQ

Metric convention: ``cosine`` is implemented as inner product on L2-normalized
inputs. Set ``normalize=True`` (the default) to have the class do it for you.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import faiss
import numpy as np


@dataclass
class IVFOPQPQConfig:
    """Configuration for the IVF-OPQ-PQ index.

    Attributes
    ----------
    dim
        Embedding dimensionality. Must equal the input vector dim.
    nlist
        Number of IVF cells. Rule of thumb: ``≈ sqrt(N)``.
    M
        PQ subvector count. Must divide ``dim``. Memory per vector is
    nbits
        Bits per PQ code. 8 is the sane default; 4 halves memory at
        the cost of noticeable recall drop.
    metric
        ``"cosine"`` (default) = IP on L2-normalized vectors.
    normalize
        Apply L2-normalization to inputs before feeding FAISS. Required
        for ``metric="cosine"``
    seed
        RNG seed for kmeans / OPQ initialization.
    """


    'Below are stanard values for corpus of 500k vectors of dim=256'
    dim: int
    nlist: int = 512
    M: int = 32
    nbits: int = 8
    metric: Literal["cosine", "l2", "ip"] = "l2"
    normalize: bool = True
    seed: int = 52

    def __post_init__(self) -> None:
        if self.dim % self.M != 0:
            raise ValueError(
                f"dim={self.dim} must be divisible by M={self.M} "
                f"(PQ requires equal-size subvectors)"
            )
        if self.nbits not in (4, 8):
            raise ValueError(f"nbits must be 4 or 8, got {self.nbits}")


class IVFOPQPQIndex:
    """Production index: IVF quantized with OPQ + PQ."""

    def __init__(self, cfg: IVFOPQPQConfig):
        self.cfg = cfg
        self._faiss_metric = self._resolve_metric()
        self._build_index()
        self._trained = False
        self._n_added = 0

    def _resolve_metric(self) -> int:
        return {
            "l2": faiss.METRIC_L2,
            "cosine": faiss.METRIC_INNER_PRODUCT,  # via normalization
            "ip": faiss.METRIC_INNER_PRODUCT,
        }[self.cfg.metric]

    def _build_index(self) -> None:
        opq = faiss.OPQMatrix(self.cfg.dim, self.cfg.M)
        quantizer = (
            faiss.IndexFlatL2(self.cfg.dim)
            if self._faiss_metric == faiss.METRIC_L2
            else faiss.IndexFlatIP(self.cfg.dim)
        )
        ivfpq = faiss.IndexIVFPQ(
            quantizer,
            self.cfg.dim,
            self.cfg.nlist,
            self.cfg.M,
            self.cfg.nbits,
            self._faiss_metric,
        )
        self._index = faiss.IndexPreTransform(opq, ivfpq)

    def _prepare(self, x: np.ndarray) -> np.ndarray:
        """Apply normalization + dtype coercion. Idempotent."""
        x = np.ascontiguousarray(x, dtype=np.float32)
        if self.cfg.normalize and self.cfg.metric == "cosine":
            x = x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)
        return x

    def train(self, corpus: np.ndarray) -> None:
        """Train OPQ rotation, IVF centroids, and PQ codebooks."""
        self._index.train(self._prepare(corpus))
        self._trained = True

    def add(self, corpus: np.ndarray) -> None:
        """Encode and store the corpus. Requires prior ``train()``."""
        if not self._trained:
            raise RuntimeError("Call train() before add().")
        corpus = self._prepare(corpus)
        self._index.add(corpus)
        self._n_added += len(corpus)

    def train_add(self, corpus: np.ndarray) -> None:
        """Convenience: train and add in one call."""
        corpus = self._prepare(corpus)
        self._index.train(corpus)
        self._index.add(corpus)
        self._trained = True
        self._n_added += len(corpus)

    def search(
        self,
        queries: np.ndarray,
        k: int = 10,
        nprobe: int = 16,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (distances, labels) top-k for each query.

        ``nprobe`` controls how many IVF cells to visit at query time.
        Higher is better recall, higher latency. 16 to 32 is usually good
        """
        queries = self._prepare(queries)
        self._set_nprobe(nprobe)
        return self._index.search(queries, k)

    def _set_nprobe(self, nprobe: int) -> None:
        # Reach through PreTransform wrapper to the IVFPQ inside.
        ivfpq = faiss.downcast_index(self._index.index)
        ivfpq.nprobe = int(nprobe) # type: ignore

    def save(self, path: str | Path) -> None:
        """Serialize both the FAISS index and its config.

        Writes two files:
            <path>            — the FAISS binary blob
            <path>.cfg.pkl    — the IVFOPQPQConfig
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(path))
        cfg_path = path.with_suffix(path.suffix + ".cfg.pkl")
        with open(cfg_path, "wb") as f:
            pickle.dump(self.cfg, f)

    @classmethod
    def load(cls, path: str | Path) -> "IVFOPQPQIndex":
        path = Path(path)
        cfg_path = path.with_suffix(path.suffix + ".cfg.pkl")
        with open(cfg_path, "rb") as f:
            cfg = pickle.load(f)
        obj = cls.__new__(cls)
        obj.cfg = cfg
        obj._faiss_metric = obj._resolve_metric()
        obj._index = faiss.read_index(str(path))
        obj._trained = True
        obj._n_added = obj._index.ntotal
        return obj

    @property
    def n_vectors(self) -> int:
        return int(self._n_added)

    @property
    def bytes_per_vector(self) -> int:
        """Compressed footprint per vector (PQ code size)."""
        return self.cfg.M * self.cfg.nbits // 8

    def __repr__(self) -> str:
        return (
            f"IVFOPQPQIndex(dim={self.cfg.dim}, nlist={self.cfg.nlist}, "
            f"M={self.cfg.M}, nbits={self.cfg.nbits}, "
            f"metric='{self.cfg.metric}', n={self.n_vectors}, "
            f"bytes/vec={self.bytes_per_vector})"
        )
