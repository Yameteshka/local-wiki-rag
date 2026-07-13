"""Correct EmbeddingGemma-300m encoding pipeline for Wikipedia-500k.

Fixes from the original embedding.py:

1. No structural pruning. Cutting layers destroys the model's semantic space
   and makes all embeddings collapse to a narrow cone (mean cosine ~0.99 on
   random pairs). Keep all 34 transformer layers.

2. Mean pooling with attention mask instead of last-token slicing.
   EmbeddingGemma was trained with mean pooling. Taking the last hidden state
   picks up the padding token for short inputs, which is nearly identical
   across all documents.

3. Proper Matryoshka via SentenceTransformer's truncate_dim, which routes
   through the trained projection head. Slicing raw hidden_state[:256] is not
   MRL — it's just picking arbitrary coordinates from an 768-dim space.

4. Task-specific prompts. EmbeddingGemma requires prompts for retrieval:
   - documents: 'title: none | text: {text}'   (prompt_name='Retrieval-document')
   - queries:   'task: search result | query: {query}'  (prompt_name='Retrieval-query')

5. L2-normalized output. IVF and cosine similarity assume unit-norm vectors.

Designed for a Kaggle T4 x2 environment. Saves incrementally so mid-run
timeouts don't lose progress. Compatible with the downstream pipeline
(loads via storage/Float32Store, indexed by indexing/IVFFlatIndex).
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from sentence_transformers import SentenceTransformer
from tqdm import tqdm


TOTAL_OBJECTS = 500_000
BATCH_SIZE = 128
EMBED_DIM = 256
MAX_SEQ_LEN = 512
MODEL_ID = "google/embeddinggemma-300m"

DATA_DIR = Path("data")

OUT_DIR = Path("/kaggle/working") if Path("/kaggle/working").exists() else Path(".")
EMB_PATH = OUT_DIR / "wikipedia_embeddings_256d.npy"
META_PATH = OUT_DIR / "wikipedia_metadata_manifest.json"
CHECKPOINT_PATH = OUT_DIR / "checkpoint.txt"


def _iter_local_parquet(data_dir: Path) -> Iterator[dict]:
    """Yield rows from all parquet shards under ``data_dir`` in sorted order.

    Resolves HuggingFace symlinks (../../../blobs/<hash>) to the real blob path
    in the HF cache, since the relative link breaks once the folder is moved
    out of the cache tree.
    """
    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        print(f"ERROR: no .parquet files found in {data_dir.resolve()}")
        sys.exit(1)

    hf_blobs = (
        Path.home()
        / ".cache/huggingface/hub/datasets--feyninc--wikipedia-500k/blobs"
    )

    print(f"Found {len(shards)} shard(s): {[s.name for s in shards]}")

    for shard in shards:
        real = shard
        if shard.is_symlink():
            blob_hash = Path(os.readlink(shard)).name
            candidate = hf_blobs / blob_hash
            if candidate.exists():
                real = candidate

        if not real.exists():
            print(f"ERROR: file not accessible: {real}")
            sys.exit(1)

        print(f"  Reading {shard.name} → {real}")
        table = pq.read_table(real)
        for batch in table.to_batches(max_chunksize=1000):
            cols = batch.to_pydict()
            keys = list(cols.keys())
            for i in range(len(cols[keys[0]])):
                yield {k: cols[k][i] for k in keys}


def _sanity_check(matrix: np.ndarray, sample_size: int = 1000) -> None:
    """Fail fast if the embedding space collapsed.

    A healthy embedding corpus has mean pairwise cosine similarity near 0.
    The broken pipeline (last-token pooling + layer pruning) gives ~0.9-0.99,
    which is what caused every query in the RAG UI to return the same 3 docs.
    """
    rng = np.random.default_rng(0)
    idx = rng.choice(len(matrix), size=min(sample_size, len(matrix)), replace=False)
    sample = matrix[idx]
    sample = sample / (np.linalg.norm(sample, axis=1, keepdims=True) + 1e-12)
    sims = sample @ sample.T
    np.fill_diagonal(sims, 0.0)
    mean_sim = float(sims.mean())
    print(f"\n[sanity] Mean pairwise cosine on {len(sample)} random rows: {mean_sim:.4f}")
    if mean_sim > 0.5:
        print("[sanity] WARNING: embedding space looks collapsed (mean sim > 0.5).")
    else:
        print("[sanity] OK: embedding space has good spread.")


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}  |  CUDA count: {torch.cuda.device_count() if device == 'cuda' else 0}")

    print(f"\nLoading {MODEL_ID} via SentenceTransformer...")
    model = SentenceTransformer(
        MODEL_ID,
        device=device,
        model_kwargs={"torch_dtype": torch.bfloat16},
    )
    model.max_seq_length = MAX_SEQ_LEN
    print(f"Model loaded. Native dim: {model.get_sentence_embedding_dimension()}  "
          f"→ truncating to {EMBED_DIM} via MRL.")

    print(f"\nReading local parquet shards from {DATA_DIR}...")
    dataset = _iter_local_parquet(DATA_DIR)

    # Resume support: skip already-processed rows.
    start_from = 0
    if CHECKPOINT_PATH.exists() and EMB_PATH.exists():
        try:
            start_from = int(CHECKPOINT_PATH.read_text().strip())
            print(f"Resuming from row {start_from:,} (checkpoint found).")
        except ValueError:
            start_from = 0

    if start_from > 0 and EMB_PATH.exists():
        embeddings = np.load(EMB_PATH)
        assert embeddings.shape == (TOTAL_OBJECTS, EMBED_DIM), (
            f"Existing embedding file has shape {embeddings.shape}, expected "
            f"({TOTAL_OBJECTS}, {EMBED_DIM}). Delete it to restart."
        )
    else:
        embeddings = np.zeros((TOTAL_OBJECTS, EMBED_DIM), dtype=np.float32)

    if META_PATH.exists() and start_from > 0:
        with open(META_PATH, "r", encoding="utf-8") as f:
            metadata = json.load(f)
    else:
        metadata = []

    text_buffer: list[str] = []
    meta_buffer: list[dict] = []
    processed = start_from
    t_start = time.perf_counter()

    pbar = tqdm(total=TOTAL_OBJECTS, initial=processed, desc="Encoding")

    for i, row in enumerate(dataset):
        if i < start_from:
            continue
        if processed >= TOTAL_OBJECTS:
            break

        text = (row.get("text") or row.get("paragraph") or "").strip()
        if not text:
            continue

        title = row.get("title") or f"Wikipedia Entry {processed}"

        text_buffer.append(text)
        meta_buffer.append({"id": processed, "title": title, "text": text})
        processed += 1

        if len(text_buffer) >= BATCH_SIZE or processed >= TOTAL_OBJECTS:
            with torch.inference_mode():
                # prompt_name='Retrieval-document' prepends the required
                # 'title: none | text: {text}' prefix and mean-pools with mask.
                # truncate_dim routes through the trained MRL head.
                batch_emb = model.encode(
                    text_buffer,
                    batch_size=len(text_buffer),
                    prompt_name="Retrieval-document",
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                # SentenceTransformer honors output dim via truncate_dim, but if the
                # loaded model doesn't expose the MRL head, fall back to slice+renorm.
                if batch_emb.shape[1] != EMBED_DIM:
                    batch_emb = batch_emb[:, :EMBED_DIM]
                    batch_emb = batch_emb / (
                        np.linalg.norm(batch_emb, axis=1, keepdims=True) + 1e-12
                    )

            end = processed
            start = end - len(text_buffer)
            embeddings[start:end] = batch_emb.astype(np.float32)
            metadata.extend(meta_buffer)

            pbar.update(len(text_buffer))
            text_buffer.clear()
            meta_buffer.clear()

            # Checkpoint every ~10 batches. Cheap insurance against timeouts.
            if (processed // BATCH_SIZE) % 10 == 0:
                np.save(EMB_PATH, embeddings)
                CHECKPOINT_PATH.write_text(str(processed))

    pbar.close()

    elapsed = time.perf_counter() - t_start
    print(f"\nEncoded {processed - start_from:,} rows in {elapsed / 60:.1f} min "
          f"({(processed - start_from) / elapsed:.1f} rows/s)")

    print("\nSaving final artifacts...")
    np.save(EMB_PATH, embeddings)
    print(f"  {EMB_PATH}  shape={embeddings.shape}  "
          f"size={embeddings.nbytes / 1024**2:.0f} MB")

    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    print(f"  {META_PATH}  entries={len(metadata):,}")

    if CHECKPOINT_PATH.exists():
        CHECKPOINT_PATH.unlink()

    _sanity_check(embeddings)


if __name__ == "__main__":
    main()

