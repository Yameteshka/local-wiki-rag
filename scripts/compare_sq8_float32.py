"""Direct comparison of SQ8 vs Float32 recall and latency.

Assumes ``scripts/build_storage.py`` has already been run.
"""

import os
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

sys.path.append(str(Path(__file__).parent.parent))

from search import (
    BruteForceSearch,
    benchmark_engine,
    compute_ground_truth,
    generate_corpus_queries,
    print_summary,
)
from storage import Float32Store, SQ8Store

TOP_K = 5
N_QUERIES = 50


def main() -> None:
    import numpy as np
    
    print("[1/3] Loading storage backends...")
    f32 = Float32Store()
    f32.load()
    f32_corpus = f32.data
    assert f32_corpus is not None
    N, D = f32_corpus.shape

    sq8 = SQ8Store()
    sq8.load()
    # Dequantize all SQ8 vectors back to float32
    all_indices = np.arange(N)
    sq8_corpus = sq8.get_vectors(all_indices)
    assert sq8_corpus is not None

    print(f"      Corpus: {N:,} × {D}")
    print(f"      Float32 footprint: {f32.get_memory_footprint():.1f}MB")
    print(f"      SQ8 footprint: {sq8.get_memory_footprint():.1f}MB")
    print(f"      Compression ratio: {f32.get_memory_footprint() / sq8.get_memory_footprint():.1f}x")

    print("\n[2/3] Preparing ground truth (BruteForce on Float32)...")
    brute_f32 = BruteForceSearch(f32_corpus)
    queries = generate_corpus_queries(f32_corpus, N_QUERIES)
    ground_truth = compute_ground_truth(queries, brute_f32, TOP_K)

    print("\n[3/3] Benchmarking...")
    results = [
        benchmark_engine(
            "Float32", brute_f32, queries, ground_truth, TOP_K,
            corpus_mb=f32.get_memory_footprint(),
        ),
        benchmark_engine(
            "SQ8", BruteForceSearch(sq8_corpus), queries, ground_truth, TOP_K,
            corpus_mb=sq8.get_memory_footprint(),
        ),
    ]
    print_summary(results)


if __name__ == "__main__":
    main()
