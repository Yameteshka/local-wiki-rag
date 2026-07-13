import sys, os, time
import psutil
import numpy as np
import torch
import torch.nn.functional as F
import faiss
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")
if device == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")


N_CENTROIDS    = 512
N_PROBE        = 64
TOP_K          = 5
N_TEST_QUERIES = 50
N_SYNTH        = 500_000
DIM            = 256


try:
    from storage import Float32Store
    f32_store = Float32Store()
    f32_store.load()   # reads storage_data/store_float32.npy
    corpus_f32 = f32_store.data.astype(np.float32)
    N, D = corpus_f32.shape
    float32_mb = f32_store.get_memory_footprint()
    print(f"Float32Store: {N:,} × {D} | {float32_mb:.1f} MB")
except Exception as e:
    print(f"Float32Store not available ({e}) — using synthetic data")
    rng = np.random.default_rng(42)
    corpus_f32 = rng.standard_normal((N_SYNTH, DIM)).astype(np.float32)
    N, D = corpus_f32.shape
    float32_mb = N * D * 4 / 1024**2

# L2-normalise once upfront so that dot product == cosine similarity at search time
corpus_f32 /= np.linalg.norm(corpus_f32, axis=1, keepdims=True)

# SQ8Store is 4× smaller — used in ANN stage 2 for memory-efficient candidate retrieval
try:
    from storage import SQ8Store
    sq8_store = SQ8Store()
    sq8_store.load()   # reads storage_data/store_sq8.npz
    sq8_mb = sq8_store.get_memory_footprint()
    print(f"SQ8Store: {sq8_store.quantized_data.shape[0]:,} × {sq8_store.quantized_data.shape[1]} | {sq8_mb:.1f} MB")
except Exception as e:
    print(f"SQ8Store not available ({e}) — ANN stage 2 will use float32")
    sq8_store = None
    sq8_mb = N * D / 1024**2   # 1 byte per element (uint8)


from indexing.ivf_index import IVFFlatConfig, IVFFlatIndex
from indexing.hnsw_index import HNSWConfig, HNSWIndex

cfg = IVFFlatConfig(
    dim=D,
    n_list=N_CENTROIDS,
    nprobe=N_PROBE,
    metric="cosine",
)

# cfg = HNSWConfig(
#     dim=D,
#     M = 32, 
#     ef_construction=160, 
#     metric="cosine"
# )

person4_index = IVFFlatIndex(cfg)
# person4_index = HNSWIndex(cfg)

print(f"Training ivf ({N_CENTROIDS} centroids) on {N:,} vectors...")
t0 = time.perf_counter()
person4_index.train_add(corpus_f32)
print(f"Done in {time.perf_counter() - t0:.1f}s")


class BruteForceSearch:
    def __init__(self, corpus, device):
        self.gpu = torch.from_numpy(corpus).to(device)   # entire corpus lives on GPU
        self.device = device

    def search(self, query: np.ndarray, top_k=TOP_K):
        q = torch.from_numpy(query.astype(np.float32)).to(self.device)
        q = F.normalize(q, dim=0)
        scores = self.gpu @ q   # (N, D) @ (D,) → (N,)
        vals, idx = torch.topk(scores, k=top_k)
        return idx.cpu().numpy(), vals.cpu().numpy()


class ANNSearch:
    """Two-stage ANN. Works with both IVFFlatIndex and HNSWIndex.

    Stage 1 pulls candidates from the underlying index; the search kwargs are
    dispatched by index type (IVF uses ``nprobe``, HNSW uses ``ef_search``).
    Stage 2 always re-ranks against the exact (SQ8-dequantised) vectors so the
    final scores are true cosine similarities regardless of the index type.
    """

    def __init__(self, index, sq8_store, corpus_fallback=None):
        self.ivf      = index
        self.sq8      = sq8_store
        self.fallback = corpus_fallback   # used when SQ8Store is not available
        self._is_hnsw = type(index).__name__ == "HNSWIndex"

    def search(self, query: np.ndarray, top_k=TOP_K, n_probe=N_PROBE):
        q = query.astype(np.float32)
        q /= np.linalg.norm(q) + 1e-12

        # Stage 1: index-specific candidate generation.
        if self._is_hnsw:
            # HNSW returns exact top-K directly — pull 4× top_k for a light rerank margin.
            n_cands = min(N, max(top_k * 4, 32))
            _, cand_ids = self.ivf.search(q.reshape(1, -1), k=n_cands)
        else:
            # IVF: request 2× the average cluster size to cover all candidates in visited cells.
            n_cands = min(N, n_probe * (N // N_CENTROIDS) * 2)
            _, cand_ids = self.ivf.search(q.reshape(1, -1), k=n_cands, nprobe=n_probe)

        cand_ids = cand_ids[0][cand_ids[0] >= 0]   # FAISS pads short results with -1

        # Stage 2: exact cosine similarity, either from SQ8Store or fallback float32.
        vecs = self.sq8.get_vectors(cand_ids) if self.sq8 else self.fallback[cand_ids]
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        scores = vecs @ q

        k = min(top_k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top])]
        return cand_ids[top], scores[top]


brute = BruteForceSearch(corpus_f32, device)
ann   = ANNSearch(person4_index, sq8_store, corpus_fallback=corpus_f32)

# warm-up: first CUDA call is slow due to kernel compilation, don't count it
dummy = np.random.randn(D).astype(np.float32)
brute.search(dummy); ann.search(dummy)
if device == "cuda":
    torch.cuda.synchronize()
print("Both engines ready")


rng = np.random.default_rng(123)
test_queries = []

# Queries must come from the corpus — random vectors are out-of-distribution
# and land on cluster boundaries, giving artificially low recall.
query_indices = rng.choice(N, size=N_TEST_QUERIES, replace=False)
for idx in query_indices:
    q = corpus_f32[idx].copy()
    norm = np.linalg.norm(q)
    if norm > 0:
        q /= norm
    # ground truth: exact brute-force top-K — this is what ANN is measured against
    gt_idx, _ = brute.search(q, top_k=TOP_K)
    test_queries.append({"query": q, "gt": set(gt_idx.tolist())})

if device == "cuda":
    torch.cuda.synchronize()
print(f"{len(test_queries)} queries ready | ground truth = exact top-{TOP_K}")


def run_benchmark(engine, corpus_mb: float) -> dict:
    latencies, recalls = [], []

    for s in test_queries:
        t0 = time.perf_counter()
        pred, _ = engine.search(s["query"], top_k=TOP_K)
        if device == "cuda":
            torch.cuda.synchronize()   # wait for GPU before stopping the clock
        latencies.append((time.perf_counter() - t0) * 1000)

        # recall = fraction of true top-K neighbours that were found
        recalls.append(len(set(pred.tolist()) & s["gt"]) / len(s["gt"]))

    proc = psutil.Process(os.getpid())
    return {
        "lat_mean" : float(np.mean(latencies)),
        "lat_p50"  : float(np.percentile(latencies, 50)),
        "lat_p95"  : float(np.percentile(latencies, 95)),
        "recall"   : float(np.mean(recalls)),
        "corpus_mb": corpus_mb,
        "ram_mb"   : proc.memory_info().rss / 1024**2,
        "lats"     : latencies,   # raw per-query latencies for the box plot
    }


print("Benchmarking...")
br = run_benchmark(brute, float32_mb)   # Iteration 1
ar = run_benchmark(ann,   sq8_mb)       # Iteration 2

for label, r in [("Brute-Force (Iter 1)", br), ("ANN / IVF  (Iter 2)", ar)]:
    print(f"\n── {label}")
    print(f"   Latency mean / p50 / p95 : {r['lat_mean']:.2f} / {r['lat_p50']:.2f} / {r['lat_p95']:.2f} ms")
    print(f"   Recall@{TOP_K}                 : {r['recall']:.3f}")
    print(f"   Corpus size               : {r['corpus_mb']:.1f} MB")


labels = ["Brute-Force\n(Iter 1)", "ANN / IVF\n(Iter 2)"]
colors = ["#4e79a7", "#f28e2b"]
FS = 12

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Search Algorithm Comparison", fontsize=14, fontweight="bold")

def bar_chart(ax, values, ylabel, title, fmt=".2f", note=""):
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor="white")
    ax.set_ylabel(ylabel, fontsize=FS)
    ax.set_title(title, fontsize=FS)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() * 1.02,
                f"{v:{fmt}}", ha="center", va="bottom", fontsize=FS - 1)
    ax.set_ylim(0, max(values) * 1.3)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    if note:
        ax.set_xlabel(note, fontsize=FS - 1, color="#555")

bar_chart(axes[0], [br["lat_mean"], ar["lat_mean"]], "ms / query", "Latency (mean)")
bar_chart(axes[1], [br["recall"],   ar["recall"]],   f"Recall@{TOP_K}", f"Recall@{TOP_K}", fmt=".3f")

reduction = br["corpus_mb"] / ar["corpus_mb"] if ar["corpus_mb"] > 0 else 1
bar_chart(axes[2], [br["corpus_mb"], ar["corpus_mb"]], "MB", "Corpus Memory",
          fmt=".0f", note=f"SQ8 is {reduction:.1f}× smaller")

plt.tight_layout()
plt.savefig("benchmark_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved benchmark_comparison.png")

fig, ax = plt.subplots(figsize=(7, 4))
bp = ax.boxplot([br["lats"], ar["lats"]], tick_labels=labels, patch_artist=True,
                medianprops=dict(color="black", linewidth=2))
for patch, color in zip(bp["boxes"], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
ax.set_ylabel("Latency (ms)", fontsize=FS)
ax.set_title(f"Latency distribution ({N_TEST_QUERIES} queries)", fontsize=FS)
ax.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.savefig("latency_distribution.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved latency_distribution.png")


speedup   = br["lat_mean"] / ar["lat_mean"] if ar["lat_mean"] > 0 else 1
mem_ratio = br["corpus_mb"] / ar["corpus_mb"] if ar["corpus_mb"] > 0 else 1

print("=" * 55)
print("VALIDATION SUMMARY")
print("=" * 55)
print(f"{'Metric':<22} {'Brute-Force':>12} {'ANN/IVF':>12}")
print("-" * 55)
rows = [
    ("Latency mean (ms)",  f"{br['lat_mean']:.2f}",    f"{ar['lat_mean']:.2f}"),
    ("Latency p50  (ms)",  f"{br['lat_p50']:.2f}",     f"{ar['lat_p50']:.2f}"),
    ("Latency p95  (ms)",  f"{br['lat_p95']:.2f}",     f"{ar['lat_p95']:.2f}"),
    (f"Recall@{TOP_K}",    f"{br['recall']:.3f}",       f"{ar['recall']:.3f}"),
    ("Corpus size  (MB)",  f"{br['corpus_mb']:.1f}",    f"{ar['corpus_mb']:.1f}"),
]
for lbl, b, a in rows:
    print(f"{lbl:<22} {b:>12} {a:>12}")
print("-" * 55)
print(f"{'Speedup':<22} {'1.00×':>12} {speedup:>11.2f}×")
print(f"{'Memory reduction':<22} {'1.00×':>12} {mem_ratio:>11.2f}×")
print("=" * 55)
print(f"Corpus: {N:,} vecs × {D}d | nprobe={N_PROBE}/{N_CENTROIDS} | device={device}")