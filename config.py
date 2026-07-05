from pathlib import Path

# --- Paths ---
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
STORAGE_DIR = PROJECT_ROOT / "storage_data"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)

# --- Data Constants ---
RAW_EMBEDDINGS_PATH = DATA_DIR / "wikipedia_embeddings_256d.npy"

# --- Vector Constants ---
NUM_VECTORS = 500_000
DIMENSIONS = 256

# --- Storage Paths ---
FLOAT32_STORE_PATH = STORAGE_DIR / "store_float32.npy"
SQ8_STORE_PATH = STORAGE_DIR / "store_sq8.npz"
