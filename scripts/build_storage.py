import sys
import time
import numpy as np
from pathlib import Path
import tracemalloc

# Adding project root to path to allow imports
sys.path.append(str(Path(__file__).parent.parent))

from pathlib import Path
from config import RAW_EMBEDDINGS_PATH, NUM_VECTORS, DIMENSIONS
from storage import Float32Store, SQ8Store

def main():
    print("="*50)
    print("STORAGE ENGINE: BUILDING DATABASE")
    print("="*50)

    # 1. Load raw embeddings
    print(f"\n[1/4] Loading raw embeddings from {RAW_EMBEDDINGS_PATH}...")
    if not RAW_EMBEDDINGS_PATH.exists():
        print(f"ERROR: File not found at {RAW_EMBEDDINGS_PATH}")
        sys.exit(1)

    start_load = time.time()
    raw_embeddings = np.load(RAW_EMBEDDINGS_PATH)
    print(f"Loaded in {time.time() - start_load:.2f}s. Shape: {raw_embeddings.shape}")

    if raw_embeddings.shape != (NUM_VECTORS, DIMENSIONS):
        print(f"WARNING: Expected shape ({NUM_VECTORS}, {DIMENSIONS}), got {raw_embeddings.shape}")

    # 2. Build Iteration 1 (Float32)
    print("\n[2/4] Building Iteration 1: Float32 Store...")
    start_f32 = time.time()
    f32_store = Float32Store()
    f32_store.build(raw_embeddings)
    print(f"Built in {time.time() - start_f32:.2f}s")

    # 3. Build Iteration 2 (SQ8)
    print("\n[3/4] Building Iteration 2: SQ8 Store...")
    start_sq8 = time.time()
    sq8_store = SQ8Store()
    sq8_store.build(raw_embeddings)
    print(f"Built in {time.time() - start_sq8:.2f}s")

    # 4. Evaluate reconstruction quality
    print("\n[4/5] Evaluating SQ8 reconstruction quality...")

    # Random sample 
    sample_size = 10_000
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(raw_embeddings.shape[0], sample_size, replace=False)

    # Original vectors
    original = raw_embeddings[sample_indices].astype(np.float32)

    # Dequantized vectors
    reconstructed = sq8_store.get_vectors(sample_indices)

    # Mean Squared Error (MSE)
    mse = np.mean((original - reconstructed) ** 2)

    # Average Cosine Similarity
    orig_norm = original / np.linalg.norm(original, axis=1, keepdims=True)
    recon_norm = reconstructed / np.linalg.norm(reconstructed, axis=1, keepdims=True)

    avg_cosine = np.mean(np.sum(orig_norm * recon_norm, axis=1))

    print(f"MSE: {mse:.8f}")
    print(f"Average Cosine Similarity: {avg_cosine:.6f}")

    # 5. Graph Quantization error
    import matplotlib.pyplot as plt

    errors = (reconstructed - original).ravel()

    print(f"Mean error: {errors.mean():.8f}")
    print(f"Std error:  {errors.std():.8f}")
    print(f"Max abs error: {np.abs(errors).max():.8f}")

    plt.figure(figsize=(8, 5))
    plt.hist(errors, bins=100)
    plt.title("SQ8 Quantization Error Distribution")
    plt.xlabel("Reconstruction Error")
    plt.ylabel("Frequency")
    plt.grid(True)

    plt.tight_layout()
    plt.show()
    plt.savefig("sq8_error_distribution.png", dpi=300)

    # -------------------------------
    # Per-dimension MSE
    # -------------------------------
    per_dim_mse = np.mean((original - reconstructed) ** 2, axis=0)

    print(f"Average per-dimension MSE: {per_dim_mse.mean():.8f}")
    print(f"Worst dimension: {np.argmax(per_dim_mse)}")
    print(f"Worst MSE: {per_dim_mse.max():.8f}")

    plt.figure(figsize=(10,5))
    plt.plot(per_dim_mse)
    plt.title("Per-Dimension Reconstruction Error (MSE)")
    plt.xlabel("Embedding Dimension")
    plt.ylabel("MSE")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("sq8_per_dimension_mse.png", dpi=300)
    plt.show()

    # -------------------------------
    # Value distribution
    # -------------------------------

    plt.figure(figsize=(10,5))

    plt.hist(
        original.ravel(),
        bins=100,
        alpha=0.6,
        density=True,
        label="Original float32"
    )

    plt.hist(
        sq8_store.quantized_data.ravel(),
        bins=100,
        alpha=0.6,
        density=True,
        label="Quantized uint8"
    )

    plt.title("Value Distribution Before and After SQ8")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.legend()

    plt.tight_layout()
    plt.savefig("sq8_value_distribution.png", dpi=300)
    plt.show()

    # -------------------------------
    # Retrieval throughput benchmark
    # -------------------------------

    print("\n[Storage Throughput Benchmark]")

    sizes = [100, 1000, 10000]
    repeats = 100

    rng = np.random.default_rng(42)

    for n in sizes:

        indices = rng.choice(NUM_VECTORS, size=n, replace=False)

        # warm-up
        sq8_store.get_vectors(indices)

        start = time.perf_counter()

        for _ in range(repeats):
            sq8_store.get_vectors(indices)

        elapsed = time.perf_counter() - start

        avg_ms = elapsed / repeats * 1000

        throughput = n / (elapsed / repeats)

        print(
            f"{n:5d} vectors | "
            f"{avg_ms:8.3f} ms | "
            f"{throughput:,.0f} vectors/sec"
        )

    # 5. Summary
    print("\n[5/5] Storage Build Summary:")
    print("-" * 30)
    print(f"Float32 RAM Footprint: {f32_store.get_memory_footprint():.2f} MB")
    print(f"SQ8 RAM Footprint:     {sq8_store.get_memory_footprint():.2f} MB")
    
    compression_ratio = f32_store.get_memory_footprint() / sq8_store.get_memory_footprint()
    print(f"Memory Reduction:      {compression_ratio:.2f}x")
    print("="*50)
    print("Storage build complete!")

if __name__ == "__main__":
    tracemalloc.start()
    main()
    current, peak = tracemalloc.get_traced_memory()
    print(f"Current memory: {current / 10**6:.2f} MB")
    print(f"Peak memory: {peak / 10**6:.2f} MB")
