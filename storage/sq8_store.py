import numpy as np
from .base_store import BaseVectorStore
from config import SQ8_STORE_PATH

class SQ8Store(BaseVectorStore):
    """
    Iteration 2: Custom Scalar Quantization (SQ8).
    Compresses float32 to uint8 (8-bit integers) per dimension.
    Reduces memory footprint by exactly 4x.
    """

    def __init__(self):
        self.quantized_data: np.ndarray | None = None  # uint8
        self.min_vals: np.ndarray | None = None        # float32 (per dimension)
        self.steps: np.ndarray | None = None           # float32 (per dimension)

    def build(self, raw_embeddings: np.ndarray) -> None:
        print(f"[SQ8Store] Quantizing data to uint8...")
        data = raw_embeddings.astype(np.float32)

        self.min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)

        ranges = max_vals - self.min_vals
        self.steps = np.where(ranges == 0, 1.0, ranges / 255.0)

        normalized = (data - self.min_vals) / self.steps
        self.quantized_data = np.clip(np.round(normalized), 0, 255).astype(np.uint8)

        print(f"[SQ8Store] Saving to {SQ8_STORE_PATH}...")
        np.savez(
            SQ8_STORE_PATH,
            quantized=self.quantized_data,
            min_vals=self.min_vals,
            steps=self.steps,
        )
        print(f"[SQ8Store] Build complete. Size: {self.get_memory_footprint():.2f} MB")

    def load(self) -> None:
        print(f"[SQ8Store] Loading from {SQ8_STORE_PATH}...")
        loaded = np.load(SQ8_STORE_PATH)
        quantized: np.ndarray = loaded['quantized']
        self.quantized_data = quantized
        self.min_vals = loaded['min_vals']
        self.steps = loaded['steps']
        print(f"[SQ8Store] Loaded. Shape: {quantized.shape}")

    def get_vectors(self, indices: np.ndarray) -> np.ndarray | None:
        """Dequantizes uint8 back to float32: original ≈ (quantized * step) + min."""
        if self.quantized_data is not None and self.min_vals is not None and self.steps is not None:
            q_vectors = self.quantized_data[indices]
            return (q_vectors.astype(np.float32) * self.steps) + self.min_vals
        return None

    def get_memory_footprint(self) -> float:
        if self.quantized_data is None:
            return 0.0
        return self.quantized_data.nbytes / (1024 ** 2)