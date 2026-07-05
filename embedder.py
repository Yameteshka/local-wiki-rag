import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_ID = "google/embeddinggemma-300m"

print("Loading EmbeddingGemma-300M (BF16, 18 Layers)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(
    MODEL_ID,
    dtype=torch.bfloat16,
).to(DEVICE)
model.eval()

model.layers = model.layers[:18]


def get_embedding(text: str) -> np.ndarray:
    if not text.strip():
        return np.zeros(256, dtype=np.float32)

    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=256,
        return_tensors="pt"
    ).to(DEVICE)

    with torch.no_grad():
        outputs = model(**inputs)
        raw_vector = outputs.last_hidden_state[0, -1, :].float().cpu().numpy()

        matryoshka_vector = raw_vector[:256]

    return matryoshka_vector