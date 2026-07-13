"""Build wikipedia.db from local parquet files in data_wiki/.

Creates a SQLite database with:
    articles_meta  — lookup by id (matches embedding row index)
    articles       — fts5 virtual table for BM25 keyword search

Run once before starting main.py:
    python scripts/build_db.py
"""

import os
import sqlite3
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "wikipedia.db"
DATA_DIR = PROJECT_ROOT / "data_wiki"
print(DATA_DIR)
TOTAL = 500_000
BATCH_COMMIT = 10_000


def iter_parquet_rows(data_dir: Path):
    """Yield dicts from all parquet shards in sorted order, resolving symlinks."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        print("ERROR: pyarrow not installed. Run: pip install pyarrow")
        sys.exit(1)

    shards = sorted(data_dir.glob("*.parquet"))
    if not shards:
        print(f"ERROR: No .parquet files found in {data_dir}")
        sys.exit(1)

    print(f"Found {len(shards)} shard(s): {[s.name for s in shards]}")

    # Blobs live in the HF cache; symlinks in data_wiki/ point there with a
    # relative path that breaks when the folder is moved out of the cache tree.
    # Extract the blob hash from the symlink target and look it up directly.
    hf_blobs = (
        Path.home()
        / ".cache/huggingface/hub/datasets--feyninc--wikipedia-500k/blobs"
    )

    for shard in shards:
        if shard.is_symlink():
            blob_hash = Path(os.readlink(shard)).name
            real = hf_blobs / blob_hash
        else:
            real = shard

        if not real.exists():
            print(f"ERROR: blob not found at {real}")
            sys.exit(1)

        print(f"  Reading {shard.name} → {real}")
        table = pq.read_table(real)
        for batch in table.to_batches(max_chunksize=1000):
            rows = batch.to_pydict()
            keys = list(rows.keys())
            for i in range(len(rows[keys[0]])):
                yield {k: rows[k][i] for k in keys}


def main() -> None:
    print(f"Building {DB_PATH} from {DATA_DIR} …")

    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    cur.executescript("""
        DROP TABLE IF EXISTS articles_meta;
        DROP TABLE IF EXISTS articles;
        CREATE TABLE articles_meta (
            id    INTEGER PRIMARY KEY,
            title TEXT NOT NULL,
            text  TEXT NOT NULL
        );
        CREATE VIRTUAL TABLE articles USING fts5(
            title,
            text,
            content='',
            tokenize='unicode61'
        );
    """)
    conn.commit()

    batch: list[tuple] = []
    doc_id = 0

    for row in iter_parquet_rows(DATA_DIR):
        if doc_id >= TOTAL:
            break

        title = str(row.get("title") or f"Entry {doc_id}")
        text = str(row.get("text") or row.get("paragraph") or "").strip()
        if not text:
            continue

        batch.append((doc_id, title, text))
        doc_id += 1

        if len(batch) >= BATCH_COMMIT:
            cur.executemany(
                "INSERT INTO articles_meta(id, title, text) VALUES (?, ?, ?)", batch
            )
            cur.executemany(
                "INSERT INTO articles(rowid, title, text) VALUES (?, ?, ?)", batch
            )
            conn.commit()
            print(f"  {doc_id:>7,} / {TOTAL:,}", end="\r", flush=True)
            batch.clear()

    if batch:
        cur.executemany(
            "INSERT INTO articles_meta(id, title, text) VALUES (?, ?, ?)", batch
        )
        cur.executemany(
            "INSERT INTO articles(rowid, title, text) VALUES (?, ?, ?)", batch
        )
        conn.commit()

    conn.close()
    size_mb = DB_PATH.stat().st_size / 1024 ** 2
    print(f"\nDone. {doc_id:,} articles → {DB_PATH}  ({size_mb:.0f} MB)")


if __name__ == "__main__":
    main()
