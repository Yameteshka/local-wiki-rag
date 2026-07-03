"""
Executes the validated 18-layer structural pruning configuration on
Google EmbeddingGemma-300M and exports the 256-dimensional Matryoshka
embedding matrix along with corresponding metadata for the RAG pipeline.
"""

import json
import torch
import numpy as np
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def main():
    # Constraints
    TOTAL_OBJECTS = 500000
    BATCH_SIZE = 256
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    MODEL_ID_HF = "./gemma_model"

    # Compatibility to prevent pruning crashes
    transformers.masking_utils._is_torch_greater_or_equal_than_2_6 = False
    transformers.models.gemma3.modeling_gemma3.create_causal_mask = lambda *args, **kwargs: None
    transformers.models.gemma3.modeling_gemma3.create_sliding_window_causal_mask = lambda *args, **kwargs: None

    print(f"Loading local model from directory '{MODEL_ID_HF}'...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_HF)

    model = AutoModel.from_pretrained(
        MODEL_ID_HF,
        dtype=torch.bfloat16,
        attn_implementation="eager"
    ).to(DEVICE)
    model.eval()

    # 18-layer pruning
    print("Applying validated 18-layer structural pruning configuration...")
    model.layers = model.layers[:18]

    print("Connecting to streaming dataset endpoint 'feyninc/wikipedia-500k'...")
    dataset = load_dataset("feyninc/wikipedia-500k", split="train", streaming=True)

    # Pre-allocate contiguous memory maps to avoid fragmentation during streaming
    # Shape is strictly constrained to 256 dimensions based on Matryoshka optimization
    final_embeddings_matrix = np.zeros((TOTAL_OBJECTS, 256), dtype=np.float32)
    metadata_manifest = []

    text_buffer = []
    processed_count = 0

    progress_bar = tqdm(total=TOTAL_OBJECTS, desc="Encoding Wikipedia Knowledge Base")

    for row in dataset:
        if processed_count >= TOTAL_OBJECTS:
            break

        text_content = row.get("text", row.get("paragraph", "")).strip()
        if not text_content:
            continue

        text_buffer.append(text_content)

        # Structure unified metadata reference for the UI layer and search validation
        doc_title = row.get("title", f"Wikipedia Entry {processed_count}")
        metadata_manifest.append({
            "id": processed_count,
            "title": doc_title,
            "text": text_content
        })

        # Execute batched inference sequences
        if len(text_buffer) == BATCH_SIZE or (processed_count + len(text_buffer) == TOTAL_OBJECTS):
            current_batch_size = len(text_buffer)

            inputs = tokenizer(
                text_buffer,
                padding=True,
                truncation=True,
                max_length=256,
                return_tensors="pt"
            ).to(DEVICE)

            with torch.no_grad():
                outputs = model(**inputs)
                # Slicing hidden states at the last sequence position and executing Matryoshka truncation
                raw_vectors = outputs.last_hidden_state[:, -1, :].float().cpu().numpy()
                matryoshka_vectors = raw_vectors[:, :256]

            # Map tensors directly to the pre-allocated array space
            final_embeddings_matrix[processed_count: processed_count + current_batch_size] = matryoshka_vectors

            processed_count += current_batch_size
            progress_bar.update(current_batch_size)
            text_buffer = []

    progress_bar.close()

    # Data serialization & handoff export
    print("\nSerializing final optimized artifacts for team handoff...")

    # Target: Storage Engineer and Indexing Engineer
    np.save("wikipedia_embeddings_256d.npy", final_embeddings_matrix)
    print("Exported: 'wikipedia_embeddings_256d.npy' [Shape: (500000, 256), Size: ~512 MB]")

    # Target: UI Integrator and Search Engineer
    with open("wikipedia_metadata_manifest.json", "w", encoding="utf-8") as f:
        json.dump(metadata_manifest, f, ensure_ascii=False, indent=2)
    print("Exported: 'wikipedia_metadata_manifest.json' [Unified Metadata Manifest]")

    print("\nPipeline production execution completed successfully.")


if __name__ == "__main__":
    main()
