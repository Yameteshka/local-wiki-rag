hello
to start use uv:
put wikipedia_embeddings_256d.npy in data folder

uv venv
source .venv/bin/activate or equivalent
uv sync
uv run build_storage.py
uv run build_db.py
streamlit run main.py