"""Hierarchical IVF — multi-level inverted file index.

Generalises flat IVF by clustering the corpus into a tree. Common
configurations:

    cluster_factors=[32, 16]    -> depth 2, 32 × 16 = 512 leaves
    cluster_factors=[8, 8, 8]   -> depth 3, 8 × 8 × 8 = 512 leaves

At query time we descend the tree, comparing only against centroids of
the locally active subtree. The number of centroid dot products per
query drops from ``k_total`` (flat IVF) to roughly ``sum(k_level)``:

    Flat IVF, k=512                       512 compares
    HIVF [32, 16], nprobe=[2, 4]          32 + 2*16 = 64
    HIVF [8, 8, 8], nprobe=[2, 2, 2]      8 + 2*8 + 4*8 = 56

The tradeoff is recall: at each level we commit to a small subset of
subtrees; if the right leaf was under a non-chosen parent it's lost.
The Inverted Multi-Index (Babenko & Lempitsky, 2014) is the canonical
reference for this construction.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Callable, Optional, Sequence, Union

import numpy as np

from indexing.clustering import Clusterer, KMeansConfig, SklearnMiniBatchKMeans
from indexing.index import BuildStats
from indexing.sources import EmbeddingSource


# Type alias: function that produces a fresh clusterer for a given branching
# factor and seed. Different subtrees need different ``n_clusters``, so we
# can't share a single pre-configured ``Clusterer`` instance.
ClustererFactory = Callable[[int, int], Clusterer]


def _default_clusterer_factory(k: int, seed: int) -> Clusterer:
    return SklearnMiniBatchKMeans(
        KMeansConfig(n_clusters=k, init="k-means++", seed=seed)
    )


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class HierarchicalIVFConfig:
    """Configuration for the hierarchical IVF index.

    Parameters
    ----------
    cluster_factors
        Branching factors per level. ``[32, 16]`` → 32 macro clusters,
        each split into 16 micro clusters (512 leaves total). The length
        of this list is the tree depth.
    truncate_dim
        MRL slice size. ``None`` keeps the full dimensionality.
    normalize
        L2-normalize vectors and centroids after truncation. Required for
        cosine similarity to behave correctly on the truncated subspace.
    min_split_size
        If a subtree contains fewer than this many vectors, stop
        splitting and turn it into a leaf even if ``cluster_factors``
        would suggest going deeper. Prevents degenerate single-vector
        clusters at the bottom of unbalanced trees.
    seed
        Base RNG seed. Level ``i`` uses ``seed + i`` so different levels
        don't share kmeans randomness.
    """

    cluster_factors: Sequence[int] = (32, 16)
    truncate_dim: Optional[int] = 256
    normalize: bool = True
    min_split_size: int = 16
    seed: int = 42


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------


@dataclass
class _Node:
    """Internal tree node.

    A node is either an *internal* node (has ``centroids`` and
    ``children``) or a *leaf* (has ``vec_positions``). The two cases are
    mutually exclusive; ``is_leaf`` disambiguates.
    """

    is_leaf: bool = False
    centroids: Optional[np.ndarray] = None
    children: list["_Node"] = field(default_factory=list)
    vec_positions: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.int64)
    )


# ---------------------------------------------------------------------------
# Index
# ---------------------------------------------------------------------------


class HierarchicalIVFIndex:
    """Tree-structured IVF index over dense embeddings.

    See module docstring for the cost/recall tradeoff. Build is recursive;
    query is a top-``nprobe`` descent down the tree.
    """

    def __init__(
        self,
        cfg: HierarchicalIVFConfig,
        clusterer_factory: Optional[ClustererFactory] = None,
    ):
        self.cfg = cfg
        self.clusterer_factory = clusterer_factory or _default_clusterer_factory

        self.root: Optional[_Node] = None
        self.vec_ids: Optional[np.ndarray] = None
        self.stats: BuildStats = BuildStats()

    # ------------------------------------------------------------------
    # Projection — kept local on purpose so HIVF doesn't depend on
    # IVFIndex internals. Cheap duplication, clean boundary.
    # ------------------------------------------------------------------

    def _project(self, vecs: np.ndarray) -> np.ndarray:
        if self.cfg.truncate_dim is not None:
            vecs = vecs[:, : self.cfg.truncate_dim]
        vecs = np.ascontiguousarray(vecs, dtype=np.float32)
        if self.cfg.normalize:
            norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-12
            vecs = vecs / norms
        return vecs

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self, source: EmbeddingSource) -> None:
        """Stream vectors out of ``source`` and recursively cluster."""
        t0 = time.perf_counter()

        all_ids: list[np.ndarray] = []
        all_vecs: list[np.ndarray] = []
        for ids, vecs in source.iter_batches(8192):
            all_ids.append(ids)
            all_vecs.append(self._project(vecs))
        self.vec_ids = np.concatenate(all_ids)
        vectors = np.concatenate(all_vecs)
        del all_ids, all_vecs

        positions = np.arange(len(vectors), dtype=np.int64)
        self.root = self._build_subtree(vectors, positions, depth=0)

        self.stats = BuildStats(
            n_vectors=len(vectors),
            source_dim=source.dim,
            index_dim=int(vectors.shape[1]),
            n_clusters=self._count_leaves(self.root),
            build_seconds=time.perf_counter() - t0,
            training_sample_size=len(vectors),
            clusterer_class=f"HierarchicalIVF{list(self.cfg.cluster_factors)}",
            timings={},
        )

    def _build_subtree(
        self, X_all: np.ndarray, positions: np.ndarray, depth: int
    ) -> _Node:
        node = _Node()

        # Stop: tree depth reached, or subtree too small to split.
        if (
            depth >= len(self.cfg.cluster_factors)
            or len(positions) <= self.cfg.min_split_size
        ):
            node.is_leaf = True
            node.vec_positions = positions
            return node

        k = self.cfg.cluster_factors[depth]
        # Not enough points to make ``k`` clusters — degenerate, become a leaf.
        if len(positions) < k:
            node.is_leaf = True
            node.vec_positions = positions
            return node

        X = X_all[positions]
        clusterer = self.clusterer_factory(k, self.cfg.seed + depth)
        centroids, labels = clusterer.fit(X)
        if self.cfg.normalize:
            centroids = centroids / (
                np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
            )
        node.centroids = centroids.astype(np.float32)

        for c in range(k):
            child_positions = positions[labels == c]
            node.children.append(
                self._build_subtree(X_all, child_positions, depth + 1)
            )
        return node

    def _count_leaves(self, node: _Node) -> int:
        if node.is_leaf:
            return 1
        return sum(self._count_leaves(ch) for ch in node.children)

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def get_candidates(
        self,
        query: np.ndarray,
        nprobe: Union[int, Sequence[int]] = 4,
    ) -> np.ndarray:
        """Descend the tree, picking the top-``nprobe`` at each level.

        Parameters
        ----------
        query
            Source-dim vector (will be projected through MRL slice + L2).
        nprobe
            Either an int (same value at every level) or a per-level list.
            ``[2, 4]`` for a depth-2 tree means top-2 macros, then top-4
            micros within each chosen macro — 8 leaves visited.

        Returns
        -------
        np.ndarray
            Unsorted external vec_ids. Final cosine ranking is the
            Search component's job.
        """
        if self.root is None or self.vec_ids is None:
            raise RuntimeError("Index has not been built.")

        depth = len(self.cfg.cluster_factors)
        if isinstance(nprobe, int):
            nprobe_list = [nprobe] * depth
        else:
            nprobe_list = list(nprobe)
            if len(nprobe_list) != depth:
                raise ValueError(
                    f"nprobe has length {len(nprobe_list)} but tree depth is {depth}"
                )

        q = self._project(query.reshape(1, -1))[0]
        positions = self._descend(self.root, q, nprobe_list, depth=0)
        if len(positions) == 0:
            return np.empty(0, dtype=np.int64)
        return self.vec_ids[positions]

    def _descend(
        self, node: _Node, q: np.ndarray, nprobe: list[int], depth: int
    ) -> np.ndarray:
        if node.is_leaf:
            return node.vec_positions

        sims = node.centroids @ q
        n_take = min(nprobe[depth], len(sims))
        # argpartition: cheap partial sort; we don't need full ordering.
        top = np.argpartition(-sims, n_take - 1)[:n_take]

        chunks: list[np.ndarray] = []
        for c in top:
            sub = self._descend(node.children[int(c)], q, nprobe, depth + 1)
            if len(sub):
                chunks.append(sub)
        if not chunks:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(chunks)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------

    def count_centroid_compares(self, nprobe: Union[int, Sequence[int]]) -> int:
        """Number of centroid dot products per query, ignoring leaves.

        This is the speed metric for HIVF — the query-time work scales
        with this, not with the leaf count.
        """
        depth = len(self.cfg.cluster_factors)
        if isinstance(nprobe, int):
            nprobe_list = [nprobe] * depth
        else:
            nprobe_list = list(nprobe)

        total = 0
        active = 1
        for level, k in enumerate(self.cfg.cluster_factors):
            total += active * k
            active *= min(nprobe_list[level], k)
        return total

    def leaf_sizes(self) -> np.ndarray:
        """Sizes of every leaf, for balance analysis.

        Use ``imbalance_metrics``-style stats (CV, Gini) on this array
        to compare against flat IVF.
        """
        sizes: list[int] = []

        def walk(node: _Node) -> None:
            if node.is_leaf:
                sizes.append(int(len(node.vec_positions)))
            else:
                for ch in node.children:
                    walk(ch)

        if self.root is not None:
            walk(self.root)
        return np.array(sizes, dtype=np.int64)
