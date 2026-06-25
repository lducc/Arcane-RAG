import json
import re
import sqlite3
import time
from multiprocessing import Pool
from pathlib import Path

from underthesea import word_tokenize

CHUNKS_FILE = Path("data/processed/chunks.jsonl")
DB_PATH = Path("data/processed/bm25.db")
NUM_WORKERS = 8
READ_BATCH_SIZE = 100000


def segment_text(text):
    cleaned = re.sub(r'[?.,;:!()"\'\-/%0-9]', ' ', text.lower())
    tokens = word_tokenize(cleaned)
    return " ".join(t.replace(" ", "_") for t in tokens)


def process_chunk(lines):
    results = []
    for line in lines:
        item = json.loads(line)
        segmented = segment_text(item["text"])
        results.append((
            segmented,
            item["chunk_id"],
            item["law_id"],
            item["doc_name"],
            item["article"],
            item.get("source_url", ""),
            item.get("relevant_doc", ""),
            item["relevant_article"],
        ))
    return results


def build_index():
    if DB_PATH.exists():
        DB_PATH.unlink()
        print("Removed existing BM25 DB")

    conn = sqlite3.connect(str(DB_PATH))
    conn.execute("PRAGMA synchronous = OFF")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA cache_size = -800000")

    conn.executescript("""
        CREATE VIRTUAL TABLE chunks_fts USING fts5(
            text,
            chunk_id UNINDEXED,
            law_id UNINDEXED,
            doc_name UNINDEXED,
            article UNINDEXED,
            source_url UNINDEXED,
            relevant_doc UNINDEXED,
            relevant_article UNINDEXED,
            tokenize='unicode61'
        );
    """)
    conn.commit()

    total = sum(1 for _ in CHUNKS_FILE.open())
    print(f"Total chunks: {total}")

    pool = Pool(NUM_WORKERS)
    processed = 0

    with CHUNKS_FILE.open() as f:
        while True:
            lines = []
            for _ in range(READ_BATCH_SIZE):
                line = f.readline()
                if not line:
                    break
                lines.append(line)
            if not lines:
                break

            worker_chunk_size = len(lines) // NUM_WORKERS + 1
            batches = [lines[i:i + worker_chunk_size] for i in range(0, len(lines), worker_chunk_size)]

            for result_batch in pool.map(process_chunk, batches):
                conn.executemany(
                    "INSERT INTO chunks_fts(text, chunk_id, law_id, doc_name, article, source_url, relevant_doc, relevant_article) VALUES (?,?,?,?,?,?,?,?)",
                    result_batch,
                )
                conn.commit()
                processed += len(result_batch)
                print(f"  Progress: {processed}/{total} ({processed/total*100:.0f}%)")

    pool.close()
    pool.join()

    conn.execute("INSERT INTO chunks_fts(chunks_fts) VALUES('optimize')")
    conn.commit()

    count = conn.execute("SELECT COUNT(*) FROM chunks_fts").fetchone()[0]
    print(f"Done — {count} entries indexed")
    conn.close()


if __name__ == "__main__":
    start = time.time()
    build_index()
    elapsed = time.time() - start
    print(f"Build time: {elapsed/60:.1f} minutes")
