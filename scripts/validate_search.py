"""Benchmark BruteForceSearch vs ANNSearch on the loaded corpus.

Assumes ``scripts/build_storage.py`` has already been run.
"""

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.append(str(Path(__file__).parent.parent))

from indexing import IVFOPQPQConfig, IVFOPQPQIndex, IVFFlatIndex, IVFFlatConfig
from indexing.hnsw_index import HNSWConfig, HNSWIndex
from search import (
    ANNConfig,
    ANNSearch,
    BruteForceSearch,
    HNSWSearch,
    HNSWSearchConfig,
    benchmark_engine,
    compute_ground_truth,
    generate_corpus_queries,
    generate_random_queries,
    print_summary,
)
from storage import Float32Store, SQ8Store

TOP_K = 5
N_QUERIES = 50
N_CENTROIDS = 512
N_PROBE = 32

# HNSW knobs — M controls graph degree, efSearch is the runtime recall/latency knob.
HNSW_M = 32
HNSW_EF_CONSTRUCTION = 200
HNSW_EF_SEARCH = 64


def main() -> None:
    print("[1/4] Loading storage backends...")
    f32 = Float32Store()
    f32.load()
    corpus = f32.data
    assert corpus is not None
    N, D = corpus.shape

    sq8 = SQ8Store()
    sq8.load()

    print(f"      Corpus: {N:,} × {D}  |  f32={f32.get_memory_footprint():.1f}MB  "
          f"sq8={sq8.get_memory_footprint():.1f}MB")

    print("\n[2/5] Training IVF-OPQ-PQ index (this is the ANN stage-1 backbone)...")
    # cfg = IVFOPQPQConfig(dim=D, nlist=N_CENTROIDS, M=D // 8, nbits=8, metric="cosine")
    # ivf = IVFOPQPQIndex(cfg)
    # t0 = time.perf_counter()
    # ivf.train_add(corpus)
    # print(f"      Trained in {time.perf_counter() - t0:.1f}s")

    print("\n[3/5] Building HNSW index (Iter 3 backbone)...")
    hnsw_cfg = HNSWConfig(
        dim=D,
        M=HNSW_M,
        ef_construction=HNSW_EF_CONSTRUCTION,
        ef_search=HNSW_EF_SEARCH,
        metric="cosine",
    )
    hnsw = HNSWIndex(hnsw_cfg)
    t0 = time.perf_counter()
    hnsw.train_add(corpus)
    # print(f"      Built in {time.perf_counter() - t0:.1f}s  ({hnsw!r})")

    print("\n[2.5/5] Training IVF index ...")
    cfg = IVFFlatConfig(dim=D, n_list=N_CENTROIDS, nprobe=64, metric="cosine")
    ivf_1 = IVFFlatIndex(cfg)
    t0 = time.perf_counter()
    ivf_1.train_add(corpus)
    print(f"      Trained in {time.perf_counter() - t0:.1f}s")

    print("\n[4/5] Preparing engines + ground truth...")
    brute = BruteForceSearch(corpus)
    # ann_1 = ANNSearch(
    #     ivf_index=ivf,
    #     rerank_store=sq8,
    #     n_vectors=N,
    #     config=ANNConfig(n_probe=N_PROBE, n_centroids=N_CENTROIDS),
    # )
    
    ann_2 = ANNSearch(
        ivf_index=ivf_1,
        rerank_store=sq8,
        n_vectors=N,
        config=ANNConfig(n_probe=N_PROBE, n_centroids=N_CENTROIDS),
    )

    
    hnsw_search = HNSWSearch(hnsw, HNSWSearchConfig(ef_search=HNSW_EF_SEARCH))
    # Queries must come from the corpus — random vectors are out-of-distribution
    # and land on cluster boundaries, giving artificially low recall (~0.58).
    queries = generate_corpus_queries(corpus, N_QUERIES)
    ground_truth = compute_ground_truth(queries, brute, TOP_K)

    # HNSW footprint: full float32 vectors + graph edges (~M * 4 bytes/node).
    hnsw_mb = (hnsw.bytes_per_vector * N + HNSW_M * 4 * N) / (1024 ** 2)

    print("\n[5/5] Benchmarking...")
    results = [
        benchmark_engine(
            "BruteForce (Iter 1)", brute, queries, ground_truth, TOP_K,
            corpus_mb=f32.get_memory_footprint(),
        ),
        # benchmark_engine(
        #     "ANN/IVFOPQ+SQ8 (Iter 2)", ann_1, queries, ground_truth, TOP_K,
        #     corpus_mb=sq8.get_memory_footprint(),
        # ),
        benchmark_engine(
            "ANN/IVF", ann_2, queries, ground_truth, TOP_K,
            corpus_mb=sq8.get_memory_footprint(),
        ),
        benchmark_engine(
            "HNSW (Iter 3)", hnsw_search, queries, ground_truth, TOP_K,
            corpus_mb=hnsw_mb,
        ),
    ]
    print_summary(results)

    # demo_query = queries[0]
    # print("\nExample response (Iteration 2 — ANN, first query):")
    # print(ann_1.search(demo_query, top_k=TOP_K))
    # print("\nExample response (Iteration 3 — HNSW, first query):")
    # print(hnsw_search.search(demo_query, top_k=TOP_K))


if __name__ == "__main__":
    main()
