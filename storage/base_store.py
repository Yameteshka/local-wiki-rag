import numpy as np
from abc import ABC, abstractmethod


class BaseVectorStore(ABC):
    """Abstract base class for Vector Storage implementations."""

    @abstractmethod
    def build(self, raw_embeddings: np.ndarray) -> None:
        """Process raw embeddings and save to disk."""
        pass

    @abstractmethod
    def load(self) -> None:
        """Load the storage from disk into memory."""
        pass

    @abstractmethod
    def get_vectors(self, indices: np.ndarray) -> np.ndarray | None:
        """
        Retrieve specific vectors by their indices.
        MUST return float32 arrays so Search can calculate exact distances.
        Returns None if the store has not been built or loaded yet.
        """
        pass

    @abstractmethod
    def get_memory_footprint(self) -> float:
        """Returns the RAM usage of the loaded data in Megabytes (MB)."""
        pass
