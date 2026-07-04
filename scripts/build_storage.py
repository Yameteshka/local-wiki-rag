import sys
import time
import numpy as np
from pathlib import Path

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
        print("Please ensure Teammate 2 has placed the .npy file in the 'data/' folder.")
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

    # 4. Summary
    print("\n[4/4] Storage Build Summary:")
    print("-" * 30)
    print(f"Float32 RAM Footprint: {f32_store.get_memory_footprint():.2f} MB")
    print(f"SQ8 RAM Footprint:     {sq8_store.get_memory_footprint():.2f} MB")
    
    compression_ratio = f32_store.get_memory_footprint() / sq8_store.get_memory_footprint()
    print(f"Memory Reduction:      {compression_ratio:.2f}x")
    print("="*50)
    print("Storage build complete!")

if __name__ == "__main__":
    main()