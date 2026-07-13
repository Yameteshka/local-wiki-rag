# Local Wiki RAG

A fully local Retrieval-Augmented Generation system over a 500K-document slice of Wikipedia. No cloud APIs, no OpenAI, no Anthropic — every component runs on your machine.

> Ask questions in natural language. Get grounded answers with cited sources. Everything stays on your hardware.

---

## What problem does this solve?

Generic LLMs hallucinate. Cloud RAG services ship your queries and private documents to third parties. Local LLMs are powerful but lack up-to-date or domain-specific knowledge.

**Local Wiki RAG** closes that gap. It bundles:

- A 500K-article Wikipedia knowledge base (precomputed 256-d Matryoshka embeddings).
- A FAISS IVF-Flat ANN index for sub-millisecond retrieval.
- A two-stage retrieval pipeline (IVF candidates → SQ8-dequantized rerank) with a BM25 fallback under low-confidence matches.
- A Streamlit chat UI that streams answers from a local Ollama LLM (`gemma3:4b`) with explicit source citations.

The result: ask *"Who was the third emperor of the Flavian dynasty?"* and get an answer like:

> The third emperor of the Flavian dynasty was Domitian... **[Source 1]** **[Source 3]**

…with clickable source cards showing the exact Wikipedia passages used.

---

## How it works

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        OFFLINE PIPELINE (run once)                      │
└─────────────────────────────────────────────────────────────────────────┘

  data_wiki/*.parquet  ──►  scripts/build_db.py        ──►  wikipedia.db
                          (SQLite FTS5 + articles_meta)      ├─ articles_meta
                                                           └─ articles (FTS5)

  EmbeddingGemma-300M  ──►  embedding_generator.py     ──►  wikipedia_embeddings_256d.npy
  (Retrieval-document         (SentenceTransformer,           (500K × 256 fp32, ~488 MB)
   prompt, MRL 256,           mean pooling, L2-norm)
   bfloat16)
                                 │
                                 ▼
                       scripts/build_storage.py
                                 │
                    ┌────────────┴───────────┐
                    ▼                        ▼
            store_float32.npy          store_sq8.npz
            (~488 MB, exact)           (~122 MB, 4× smaller)
                    │                        │
                    └───────────┬────────────┘
                                ▼
                  IVFFlatIndex (512 centroids, nprobe=64)
                  Trained at UI startup by ui/validation.py
                                │
                                ▼
                     ANNSearch (two-stage)
                     Stage 1: IVF candidate generation
                     Stage 2: SQ8 dequant + exact cosine rerank


┌─────────────────────────────────────────────────────────────────────────┐
│                          ONLINE / RUNTIME                               │
└─────────────────────────────────────────────────────────────────────────┘

   User query (Streamlit chat)
        │
        ▼
   ui/embedder.get_embedding()       EmbeddingGemma-300M
        │                            prompt="Retrieval-query"
        ▼                            → 256-d unit vector
   ANNSearch.search(q, top_k=3)
        │
        ├─ max_score < 0.30 ?  ──YES──►  SQLite FTS5 BM25 fallback
        │                                              (builtin_bm25_search)
        │
        ▼ NO
   get_metadata_by_ids()             SQLite SELECT title, text FROM articles_meta
        │
        ▼
   generate_answer()                 POST localhost:11434/api/generate
        │                            model: gemma3:4b, 8K ctx, temp 0.3
        ▼
   Streamlit renders answer + expandable "Sources Used" panel
```

---

## Key features

- **Local-first.** No external API calls at runtime. EmbeddingGemma, FAISS, SQLite, and Ollama all run on your machine.
- **Matryoshka embeddings.** EmbeddingGemma-300M's trained projection head compresses native 768-d vectors to 256-d with ~1–2% recall loss vs full dim.
- **Two-stage retrieval.** IVF-Flat narrows 500K candidates to ~60K in sub-millisecond; SQ8-dequantized rerank picks the top-3 with exact cosine. Total query latency: 1–5 ms on GPU.
- **Hybrid fallback.** When dense-retrieval confidence drops below 0.30, the system transparently retries with BM25 over SQLite FTS5 — keyword search catches what embeddings miss.
- **Cited answers.** Every LLM response includes `[Source N]` citations mapped to expandable source cards with title, similarity score, and full passage.
- **Quantized storage.** SQ8 scalar quantization reduces the 488 MB fp32 matrix to 122 MB with negligible reconstruction error.
- **Sanity-checked corpus.** The embedding generator runs a mean pairwise cosine similarity check at the end of generation. If the embedding space collapses (mean sim > 0.5), it warns immediately — the most common failure mode for compressed retrieval models.

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Embedding model | `google/embeddinggemma-300m` | Native Matryoshka, task-specific retrieval prompts, 8K context, open weights |
| Embedding framework | `sentence-transformers` | Loads the trained pooling config + MRL projection head + prompt templates — `AutoModel` alone can't |
| Vector storage | NumPy fp32 + SQ8 quantized | Two tiers: exact fp32 for ground truth, SQ8 for production (4× smaller) |
| ANN index | FAISS IVF-Flat (512 centroids, nprobe=64) | Sublinear search, exact final scoring, no PQ approximation |
| Keyword fallback | SQLite FTS5 (BM25) | Stdlib, zero extra deps, catches low-confidence queries |
| LLM | Ollama `gemma3:4b` (4B params) | Local, fast, good enough for grounded QA with retrieval context |
| UI | Streamlit | One-file chat app, instant iteration |
| Compute | PyTorch + CUDA, bfloat16 | T4/L4/3080-class GPU; CPU fallback works for low traffic |

---

## Project structure

```
local-wiki-rag/
├── main.py                    # Streamlit chat UI + orchestration (entry point)
├── embedding_generator.py     # Kaggle-targeted corpus encoder (offline)
├── convert_json_to_db.py      # Legacy JSON→SQLite importer (deprecated)
├── config.py                  # Paths and corpus dimensions
│
├── ui/
│   ├── embedder.py            # Query-time encoder (Retrieval-query prompt)
│   └── validation.py          # Runtime engine factory (IVF + SQ8 + ANN)
│
├── storage/
│   ├── base_store.py          # BaseVectorStore ABC
│   ├── float32_store.py       # Iteration 1: raw fp32 (~488 MB)
│   └── sq8_store.py           # Iteration 2: SQ8 quantized (~122 MB)
│
├── indexing/
│   ├── sources.py             # EmbeddingSource protocol
│   ├── evaluate.py            # FAISS recall@k + latency harness
│   ├── ivf_index.py           # IVF-Flat wrapper
│   ├── ivf_opq_pq_index.py    # IVF-OPQ-PQ wrapper (experimented)
│   ├── hnsw_index.py          # HNSW wrapper (experimented)
│   ├── experiments/           # CSV outputs from benchmark notebooks
│   └── notebooks/             # Indexing ablation notebooks
│
├── search/
│   ├── types.py               # Hit / SearchResult TypedDicts
│   ├── base_search.py         # BaseSearchEngine ABC
│   ├── brute_force.py         # Iteration 1: exact KNN on GPU
│   ├── ann_search.py          # Iteration 2: two-stage IVF + SQ8 rerank
│   ├── hnsw_search.py         # Iteration 3: HNSW engine
│   ├── validation.py          # Benchmark harness
│   └── notebooks/             # Search iteration notebooks
│
└── scripts/
    ├── build_db.py            # Build wikipedia.db from parquet
    ├── build_storage.py       # Build Float32 + SQ8 stores
    └── validate_search.py     # End-to-end search benchmarks
```

---

## Prerequisites

1. **Python 3.14+** (the project uses `uv` for env management)
2. **uv** — install via `curl -LsSf https://astral.sh/uv/install.sh | sh`
3. **Ollama** running locally — install from <https://ollama.com>
4. **Ollama model pulled** — `ollama pull gemma3:4b`
5. **GPU** (recommended) — NVIDIA T4 / L4 / RTX 3080-class or better with 8 GB+ VRAM. CPU-only works but expect 3–5× slower inference.
6. **Disk space** — ~3 GB for model weights, dataset shards, and produced stores/indexes.
7. **HuggingFace access** — `google/embeddinggemma-300m` is gated; run `huggingface-cli login` with a token that has access.

---

## Installation & setup

### 1. Clone and install dependencies

```bash
git clone https://github.com/Yameteshka/local-wiki-rag.git
cd local-wiki-rag

uv venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
uv sync

# Two implicit deps not yet declared in pyproject.toml
uv pip install pyarrow tqdm
```

### 2. Get the Wikipedia-500k dataset

The project expects HuggingFace `feyninc/wikipedia-500k` parquet shards under `data_wiki/`:

```bash
# Option A: download via huggingface_hub
python -c "from huggingface_hub import snapshot_download; snapshot_download('feyninc/wikipedia-500k', repo_type='dataset', local_dir='data_wiki')"

# Option B: copy from an existing HF cache and resolve symlinks
# (see scripts/build_db.py for symlink resolution logic)
```

### 3. Generate the embeddings (one-time, ~3 hours on Kaggle T4)

Run `embedding_generator.py` on a GPU machine (Kaggle T4 ×2 recommended). The script:

- Loads `google/embeddinggemma-300m` via `SentenceTransformer`
- Encodes 500K documents with `prompt_name="Retrieval-document"`, `truncate_dim=256`, `normalize_embeddings=True`
- Saves `wikipedia_embeddings_256d.npy` (~488 MB) + `wikipedia_metadata_manifest.json`
- Runs a sanity check (mean pairwise cosine sim must be < 0.5)

Place the resulting `wikipedia_embeddings_256d.npy` into `data/`.

### 4. Build the SQLite database

```bash
uv run scripts/build_db.py
```

Reads `data_wiki/*.parquet`, builds `wikipedia.db` with two tables:
- `articles_meta(id, title, text)` — metadata lookup
- `articles` (FTS5 virtual table) — BM25 keyword fallback

### 5. Build the vector stores

```bash
uv run scripts/build_storage.py
```

Reads `data/wikipedia_embeddings_256d.npy` and produces:
- `storage_data/store_float32.npy` (~488 MB, exact)
- `storage_data/store_sq8.npz` (~122 MB, 4× smaller, negligible reconstruction error)

### 6. Start Ollama

```bash
ollama serve            # in a separate terminal
ollama pull gemma3:4b   # one-time
```

### 7. Launch the UI

```bash
streamlit run main.py
```

The first launch trains the IVF-Flat index (512 centroids, ~30 seconds on CPU, ~5 seconds on GPU). Subsequent launches can cache the trained index.

Open <http://localhost:8501> and start asking questions.

---

## Usage

Type any factual question into the chat box. Examples that work well:

- *"Who painted the Mona Lisa?"*
- *"What was the cause of the Panic of 1893?"*
- *"Explain the difference between TCP and UDP."*

The assistant will respond with cited sources. Expand the **Sources Used** panel below each answer to see the exact Wikipedia passages used.

If the assistant responds *"I cannot answer this based on the provided local data"*, the dense-retrieval confidence was below 0.30 and the BM25 fallback also returned no relevant matches. Try rephrasing or asking about something covered by Wikipedia.

---

## Performance characteristics

Measured on a 500K-document corpus, RTX 3080 / Kaggle T4:

| Metric | Value |
|---|---|
| Corpus encoding (one-time) | ~3 hours on T4 ×2, bfloat16, batch=128 |
| Index training (IVF, 512 centroids) | ~5 s on GPU, ~30 s on CPU |
| Embedding storage (SQ8) | 122 MB (4× smaller than fp32) |
| SQLite DB size | ~250 MB |
| Query embedding latency | 3–5 ms (Retrieval-query prompt) |
| ANN search latency (IVF + SQ8 rerank) | 1–3 ms (top-3, nprobe=64) |
| BM25 fallback latency | ~1 ms |
| LLM generation latency | 0.5–3 s per response (gemma3:4b, 512 max tokens) |
| End-to-end wall clock | ~1–4 s per query |

---

## Roadmap & future improvements

### Retrieval quality

- **BEIR-style evaluation suite.** Currently the system relies on sanity checks (mean pairwise cosine sim) and UI smoke tests. A proper benchmark on MS-MARCO / BEIR subsets with nDCG@10 would give quantitative recall numbers and allow systematic comparison of future changes.
- **Hybrid retrieval scoring.** Instead of the binary "vector → fallback to BM25" gate, combine dense and sparse scores (e.g., Reciprocal Rank Fusion) for every query. This typically adds 3–5% recall@10 on dense+sparse ensembles.
- **Query expansion.** Pass the user query through a small LLM (or EmbeddingGemma itself with `Retrieval-query` prompt variants) to generate paraphrases, embed each, and average. Cheap recall boost.
- **Re-ranking with a cross-encoder.** Add a final cross-encoder rerank (e.g., `ms-marco-MiniLM-L-6-v2`) over the top-20 ANN candidates. Expected +5–8% nDCG@10.

### Compression & scale

- **IVF-PQ with OPQ rotation.** Explored in `indexing/notebooks/4_ivfpq_benchmark.ipynb` but not shipped. With `M=32, nbits=8`, the index drops to 32 bytes/vector (32× smaller than fp32) at ~25% recall loss — acceptable tradeoff for >5M corpora.
- **HNSW for low-latency CPU inference.** Explored in `indexing/notebooks/1_faiss_benchmark.ipynb`. HNSW (M=32, efSearch=160) reaches 99.65% recall at 0.045 ms — better than IVF for CPU-only deployments.
- **Sharded index for >5M documents.** Current IVF-Flat holds 500K × 256 fp32 in 512 MB. For 5M+ documents, shard by topic cluster and dispatch queries through a coarse pre-classifier.

### Engineering

- **Consolidate configuration.** Move Ollama URL/model, BM25 threshold, batch sizes, and IVF hyperparameters into `config.py` — currently scattered as inline constants across `main.py`, `ui/validation.py`, and `embedding_generator.py`.
- **Add `pyarrow` and `tqdm` to `pyproject.toml`.** They are transitive today; explicit declaration avoids silent breakage.
- **Soften Python 3.14 floor.** Nothing in the codebase truly requires 3.14; Python 3.11+ would work and broaden compatibility.
- **Retire `convert_json_to_db.py`.** `scripts/build_db.py` supersedes it (reads parquet directly, builds both metadata and FTS5 tables).
- **Clean up `ui/validation.py`.** The runtime engine factory is misnamed "validation" (historical: it started as a benchmark notebook). Rename to `ui/engine.py` and strip the dead matplotlib branch.
- **Add a CI workflow.** Lint, type-check, and run `scripts/validate_search.py` on a small fixture corpus to catch regressions.
- **Cache the trained IVF index to disk.** Right now `ui/validation.py` retrains on every Streamlit launch. Pickle the trained index to `storage_data/ivf_index.pkl` and load on startup.

### Features

- **Streaming LLM output.** Currently `generate_answer()` waits for the full response (`stream=False`). Switch to streaming for a more responsive UX.
- **Multi-turn context.** Track the last N turns in the Streamlit session and prepend them to the LLM prompt for follow-up questions.
- **Source highlighting.** Highlight the exact sentences in each source passage that the LLM used in the answer (via a small alignment model or attention rollout).
- **Document upload.** Let users drop in their own PDFs/Markdown and have them embedded + indexed on the fly into a separate namespace.
- **Multi-language queries.** EmbeddingGemma is multilingual; relax the "English only" system prompt and add language detection.

---

## Limitations

- **Corpus is a 500K slice of Wikipedia**, not the full 5M+ articles. Niche topics may not be covered.
- **No multi-turn dialogue memory.** Each query is independent.
- **LLM is a 4B model.** Expect weaker reasoning than GPT-4/Claude on complex multi-hop questions. Grounded single-hop QA is the sweet spot.
- **No quantitative retrieval benchmarks yet.** See Roadmap §"BEIR-style evaluation suite".
- **First Streamlit boot trains the IVF index from scratch** (~5 s on GPU, ~30 s on CPU). Caching is on the Roadmap.

---

## License

MIT License. See [LICENSE](LICENSE).

Copyright (c) 2026 Yameteshka

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Acknowledgements

- **Google** for [EmbeddingGemma-300M](https://huggingface.co/google/embeddinggemma-300m) — open-weight, MRL-trained, retrieval-tuned.
- **FAISS team (Meta)** for the [FAISS](https://github.com/facebookresearch/faiss) library.
- **feyninc** for the [wikipedia-500k](https://huggingface.co/datasets/feyninc/wikipedia-500k) dataset slice.
- **Ollama** for the local LLM runtime.
- **Streamlit** for the rapid-prototyping UI framework.
