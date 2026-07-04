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
        self.quantized_data: np.ndarray = None  # uint8
        self.min_vals: np.ndarray = None        # float32 (per dimension)
        self.steps: np.ndarray = None           # float32 (per dimension)

    def build(self, raw_embeddings: np.ndarray) -> None:
        print(f"[SQ8Store] Quantizing data to uint8...")
        data = raw_embeddings.astype(np.float32)
        
        # 1. Calculate min and max for EACH dimension (column-wise)
        self.min_vals = np.min(data, axis=0)
        max_vals = np.max(data, axis=0)
        
        # 2. Calculate step size (scale factor)
        # Prevent division by zero if a dimension has no variance
        ranges = max_vals - self.min_vals
        self.steps = np.where(ranges == 0, 1.0, ranges / 255.0)
        
        # 3. Quantize to 0-255 range and cast to uint8
        # We clip to ensure no overflow during casting
        normalized = (data - self.min_vals) / self.steps
        self.quantized_data = np.clip(np.round(normalized), 0, 255).astype(np.uint8)
        
        # 4. Save to disk
        print(f"[SQ8Store] Saving to {SQ8_STORE_PATH}...")
        np.savez(
            SQ8_STORE_PATH,
            quantized=self.quantized_data,
            min_vals=self.min_vals,
            steps=self.steps
        )
        print(f"[SQ8Store] Build complete. Size: {self.get_memory_footprint():.2f} MB")

    def load(self) -> None:
        print(f"[SQ8Store] Loading from {SQ8_STORE_PATH}...")
        loaded = np.load(SQ8_STORE_PATH)
        self.quantized_data = loaded['quantized']
        self.min_vals = loaded['min_vals']
        self.steps = loaded['steps']
        print(f"[SQ8Store] Loaded. Shape: {self.quantized_data.shape}")

    def get_vectors(self, indices: np.ndarray) -> np.ndarray:
        """
        Retrieves vectors and DEQUANTIZES them back to float32.
        This is crucial: Person 5 needs float32 to calculate exact Cosine Similarity.
        """
        # 1. Fetch the compressed uint8 vectors
        q_vectors = self.quantized_data[indices]
        
        # 2. Dequantize back to float32
        # Formula: original ≈ (quantized * step) + min
        float_vectors = (q_vectors.astype(np.float32) * self.steps) + self.min_vals
        
        return float_vectors

    def get_memory_footprint(self) -> float:
        if self.quantized_data is None:
            return 0.0
        # Only counting the main data array, min_vals and steps are negligible (~1KB)
        return self.quantized_data.nbytes / (1024 ** 2)