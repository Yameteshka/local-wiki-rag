"""Build float16 copy of embeddings and report reconstruction MSE + avg cosine.

Assumes the float32 store or raw embeddings are present (run `build_storage.py` first).
Saves `storage_data/store_float16.npy`.

on 10000 sample mse is 0 and cosine similarity is 1 of float16 -> float32 reconstruciton
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.append(str(Path(__file__).parent.parent))
from config import FLOAT32_STORE_PATH, RAW_EMBEDDINGS_PATH

STORAGE_DIR = Path(__file__).resolve().parent.parent / "storage_data"
FLOAT16_PATH = STORAGE_DIR / "store_float16.npy"


def load_embeddings() -> np.ndarray:
    if FLOAT32_STORE_PATH.exists():
        print(f"Loading float32 store from {FLOAT32_STORE_PATH}...")
        data = np.load(FLOAT32_STORE_PATH)
    elif RAW_EMBEDDINGS_PATH.exists():
        print(f"Loading raw embeddings from {RAW_EMBEDDINGS_PATH}...")
        data = np.load(RAW_EMBEDDINGS_PATH)
    else:
        raise FileNotFoundError(
            "Neither float32 store nor raw embeddings found. Run build_storage.py first."
        )
    if data.ndim != 2:
        raise ValueError(f"Expected 2D embeddings, got shape={data.shape}")
    return data.astype(np.float32, copy=False)


def compute_metrics(orig: np.ndarray, recon: np.ndarray, sample_size: Optional[int] = 10000):
    n = orig.shape[0]
    if sample_size is None or sample_size >= n:
        idx = np.arange(n)
    else:
        rng = np.random.default_rng(42)
        idx = rng.choice(n, size=sample_size, replace=False)
    o = orig[idx]
    r = recon[idx]
    mse = np.mean((o - r) ** 2)
    # cosine similarity per row
    o_norm = o / (np.linalg.norm(o, axis=1, keepdims=True) + 1e-12)
    r_norm = r / (np.linalg.norm(r, axis=1, keepdims=True) + 1e-12)
    avg_cos = float(np.mean(np.sum(o_norm * r_norm, axis=1)))
    return mse, avg_cos


def main() -> None:
    p = argparse.ArgumentParser(description="Build float16 store and report reconstruction metrics")
    p.add_argument("--sample", "-s", type=int, default=10000, help="Sample size for metrics (default 10000). Use 0 for full).")
    args = p.parse_args()

    orig = load_embeddings()
    N, D = orig.shape
    print(f"Embeddings loaded: {N:,} × {D}")

    start = time.perf_counter()
    f16 = orig.astype(np.float16)
    # Ensure storage dir exists
    STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(FLOAT16_PATH, f16)
    elapsed = time.perf_counter() - start

    recon = f16.astype(np.float32)
    sample = None if args.sample == 0 else args.sample
    mse, avg_cos = compute_metrics(orig, recon, sample_size=sample)

    filesize_mb = FLOAT16_PATH.stat().st_size / (1024 ** 2)
    mem_footprint_mb = f16.nbytes / (1024 ** 2)

    print(f"Saved float16 store: {FLOAT16_PATH} ({filesize_mb:.2f} MB) in {elapsed:.2f}s")
    print(f"In-memory float16 footprint: {mem_footprint_mb:.2f} MB")
    print(f"Reconstruction MSE (orig - float16->float32): {mse:.8f}")
    print(f"Average cosine similarity (orig vs recon): {avg_cos:.6f}")


if __name__ == '__main__':
    main()


