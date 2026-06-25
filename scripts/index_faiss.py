import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

INPUT = Path("data/processed/chunks.jsonl")
INDEX_DIR = Path("data/processed/faiss_index")
ENCODE_BATCH = 1024
DEVICE = "cuda"
MODEL_NAME = "intfloat/multilingual-e5-small"
DIM = 384


def main():
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}", file=sys.stderr)
    print(f"Loading model: {MODEL_NAME}", file=sys.stderr)
    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME, device=DEVICE)
    print(f"Model loaded in {time.time() - t0:.0f}s", file=sys.stderr)

    index = faiss.IndexFlatIP(DIM)
    index = faiss.IndexIDMap(index)

    all_payloads = []
    offsets = []
    total = 0

    t1 = time.time()
    with open(INPUT, "rb") as f:
        texts_batch = []
        while True:
            pos = f.tell()
            line = f.readline()
            if not line:
                break
            offsets.append(pos)
            item = json.loads(line.decode("utf-8"))
            all_payloads.append({k: v for k, v in item.items() if k != "text"})
            texts_batch.append("passage: " + item["text"])

            if len(texts_batch) == ENCODE_BATCH:
                vectors = model.encode(texts_batch, normalize_embeddings=True)
                vectors = vectors.astype(np.float32)
                ids = np.arange(total, total + len(vectors), dtype=np.int64)
                index.add_with_ids(vectors, ids)
                total += len(vectors)
                texts_batch = []
                if total % (ENCODE_BATCH * 10) == 0:
                    import torch
                    torch.cuda.empty_cache()
                    print(f"  Encoded {total} chunks...", file=sys.stderr)

        if texts_batch:
            vectors = model.encode(texts_batch, normalize_embeddings=True)
            vectors = vectors.astype(np.float32)
            ids = np.arange(total, total + len(vectors), dtype=np.int64)
            index.add_with_ids(vectors, ids)
            total += len(vectors)

    print(f"Encoded + indexed {total} vectors in {time.time() - t1:.0f}s", file=sys.stderr)

    faiss.write_index(index, str(INDEX_DIR / "index.faiss"))
    print(f"FAISS index saved ({index.ntotal} vectors)", file=sys.stderr)

    with open(INDEX_DIR / "payloads.pkl", "wb") as f:
        pickle.dump(all_payloads, f)
    with open(INDEX_DIR / "offsets.pkl", "wb") as f:
        pickle.dump(offsets, f)
    print(f"Payloads + offsets saved ({time.time() - t1:.0f}s)", file=sys.stderr)

    print(f"\nDone — {total} vectors indexed", file=sys.stderr)
    print(f"  index.faiss:   {Path(INDEX_DIR / 'index.faiss').stat().st_size / 1e9:.2f} GB", file=sys.stderr)
    print(f"  payloads.pkl:  {Path(INDEX_DIR / 'payloads.pkl').stat().st_size / 1e9:.2f} GB", file=sys.stderr)


if __name__ == "__main__":
    main()
