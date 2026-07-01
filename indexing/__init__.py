"""IVF indexing module for the local-wiki-rag project.

Public API:
    EmbeddingSource, MockSource, NumpyFileSource
    Clusterer, KMeansConfig, SklearnKMeans, SklearnMiniBatchKMeans, SphericalKMeans
    IVFConfig, IVFIndex
    cluster_sizes, imbalance_metrics
"""

from indexing.sources import EmbeddingSource, MockSource, NumpyFileSource
from indexing.clustering import (
    Clusterer,
    KMeansConfig,
    SklearnKMeans,
    SklearnMiniBatchKMeans,
    SphericalKMeans,
)
from indexing.index import IVFConfig, IVFIndex
from indexing.hierarchical import HierarchicalIVFConfig, HierarchicalIVFIndex
from indexing.diagnostics import cluster_sizes, compute_inertia, imbalance_metrics

__all__ = [
    "EmbeddingSource",
    "MockSource",
    "NumpyFileSource",
    "Clusterer",
    "KMeansConfig",
    "SklearnKMeans",
    "SklearnMiniBatchKMeans",
    "SphericalKMeans",
    "IVFConfig",
    "IVFIndex",
    "HierarchicalIVFConfig",
    "HierarchicalIVFIndex",
    "cluster_sizes",
    "compute_inertia",
    "imbalance_metrics",
]
