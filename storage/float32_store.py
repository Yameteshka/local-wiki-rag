import numpy as np
from .base_store import BaseVectorStore
from config import FLOAT32_STORE_PATH


class Float32Store(BaseVectorStore):
    """Iteration 1: Raw float32 storage. Baseline for comparison."""

    def __init__(self):
        self.data: np.ndarray | None = None

    def build(self, raw_embeddings: np.ndarray) -> None:
        print(f"[Float32Store] Building and saving to {FLOAT32_STORE_PATH}...")
        self.data = raw_embeddings.astype(np.float32)
        np.save(FLOAT32_STORE_PATH, self.data)
        print(f"[Float32Store] Build complete. Size: {self.get_memory_footprint():.2f} MB")

    def load(self) -> None:
        print(f"[Float32Store] Loading from {FLOAT32_STORE_PATH}...")
        data: np.ndarray = np.load(FLOAT32_STORE_PATH)
        self.data = data
        print(f"[Float32Store] Loaded. Shape: {data.shape}")

    def get_vectors(self, indices: np.ndarray) -> np.ndarray | None:
        if self.data is None:
            return None
        return self.data[indices]

    def get_memory_footprint(self) -> float:
        if self.data is None:
            return 0.0
        return self.data.nbytes / (1024 ** 2)
