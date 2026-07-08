"""Query encoder
The corpus was encoded with:
    SentenceTransformer("google/embeddinggemma-300m")
    prompt_name="Retrieval-document"
    truncate_dim=256, normalize_embeddings=True
"""

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "google/embeddinggemma-300m"
EMBED_DIM = 256
MAX_SEQ_LEN = 512

print(f"Loading EmbeddingGemma-300M on {DEVICE}...")
_model = SentenceTransformer(
    MODEL_ID,
    device=DEVICE,
    model_kwargs={"torch_dtype": torch.bfloat16},
)
_model.max_seq_length = MAX_SEQ_LEN
print(f"Ready. Native dim: {_model.get_sentence_embedding_dimension()} "
      f"→ MRL-truncated to {EMBED_DIM}.")


def get_embedding(text: str) -> np.ndarray:
    """Encode a single user query into a 256-dim unit-norm float32 vector."""
    if not text.strip():
        return np.zeros(EMBED_DIM, dtype=np.float32)

    with torch.inference_mode():
        vec = _model.encode(
            text,
            prompt_name="Retrieval-query",
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    if vec.shape[-1] != EMBED_DIM:
        vec = vec[:EMBED_DIM]
        vec = vec / (np.linalg.norm(vec) + 1e-12)

    return vec.astype(np.float32)
