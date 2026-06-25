import json
import pickle
import re
import sqlite3
from pathlib import Path

import faiss
import numpy as np
import torch
from underthesea import word_tokenize
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSequenceClassification

INDEX_DIR = Path("data/processed/faiss_index")
CHUNKS_PATH = Path("data/processed/chunks.jsonl")
DB_PATH = Path("data/processed/bm25.db")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "intfloat/multilingual-e5-small"
RERANKER_NAME = "BAAI/bge-reranker-v2-m3"

VIETNAMESE_STOPWORDS = frozenset({
    'các', 'của', 'và', 'được', 'những', 'về', 'có', 'cho', 'theo', 'tại',
    'với', 'để', 'trong', 'khi', 'là', 'này', 'đó', 'một', 'hoặc', 'không',
    'nếu', 'thì', 'qua', 'ra', 'vào', 'trên', 'xuống', 'đi', 'lại', 'đã',
    'đang', 'sẽ', 'đều', 'bị', 'do', 'việc', 'nào',
})

BM25_TOP_K = 200
DENSE_TOP_K = 100
RRF_TOP_K = 100
FINAL_TOP_K = 12
RRF_CONST = 60


def load_index():
    index = faiss.read_index(str(INDEX_DIR / "index.faiss"))
    with open(INDEX_DIR / "payloads.pkl", "rb") as file:
        payloads = pickle.load(file)
    with open(INDEX_DIR / "offsets.pkl", "rb") as file:
        offsets = pickle.load(file)
    return index, payloads, offsets


def load_model():
    return SentenceTransformer(MODEL_NAME, device=DEVICE)


def load_reranker():
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_NAME, torch_dtype=torch.float16
    )
    model = model.to(DEVICE)
    model.eval()
    return tokenizer, model


def get_database():
    connection = sqlite3.connect(str(DB_PATH))
    connection.execute("PRAGMA cache_size = -400000")
    return connection


def build_fts_query(question):
    cleaned = re.sub(r'[?.,;:!()"\'\-/%0-9]', ' ', question.lower())
    segmented = word_tokenize(cleaned)
    tokens = [t.replace(" ", "_") for t in segmented]
    terms = [t for t in tokens if t not in VIETNAMESE_STOPWORDS and len(t) >= 2]
    if not terms:
        return ""
    terms.sort(key=len, reverse=True)
    return " AND ".join(terms[:3])


def search_bm25(connection, query, top_k=BM25_TOP_K):
    fts_query = build_fts_query(query)
    if not fts_query:
        return []

    try:
        rows = connection.execute("""
            SELECT rowid, text, chunk_id, relevant_article, doc_name, article, source_url, law_id
            FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY rank LIMIT ?
        """, (fts_query, top_k)).fetchall()
    except sqlite3.OperationalError:
        return []

    return [
        {"rowid": row[0], "text": row[1], "chunk_id": row[2], "relevant_article": row[3],
         "doc_name": row[4], "article": row[5], "source_url": row[6], "law_id": row[7]}
        for row in rows
    ]


_chunk_cache = None


def _load_chunk_cache():
    global _chunk_cache
    if _chunk_cache is not None:
        return _chunk_cache
    _chunk_cache = {}
    with CHUNKS_PATH.open() as f:
        for line in f:
            item = json.loads(line)
            _chunk_cache[item["chunk_id"]] = item["text"]
    return _chunk_cache


def _dense_search_vectors(vectors, index, payloads, offsets, top_k=DENSE_TOP_K):
    scores, indices = index.search(vectors.astype(np.float32), top_k)
    cache = _load_chunk_cache()
    results = []
    seen = set()
    for row_idx in range(vectors.shape[0]):
        for i, idx in enumerate(indices[row_idx]):
            if idx == -1 or idx in seen:
                continue
            seen.add(int(idx))
            payload = payloads[idx]
            chunk_id = payload.get("chunk_id", "")
            results.append({
                "chunk_id": chunk_id,
                "relevant_article": payload.get("relevant_article", ""),
                "doc_name": payload.get("doc_name", ""),
                "article": payload.get("article", ""),
                "source_url": payload.get("source_url", ""),
                "law_id": payload.get("law_id", ""),
                "dense_score": float(scores[row_idx][i]),
                "dense_rank": len(results) + 1,
                "text": cache.get(chunk_id, ""),
            })

    return results[:top_k]


def search_dense(query, model, index, payloads, offsets, top_k=DENSE_TOP_K):
    vector = model.encode("query: " + query, normalize_embeddings=True)
    return _dense_search_vectors(vector.reshape(1, -1), index, payloads, offsets, top_k)


def search_dense_hyde(query, hyde_text, model, index, payloads, offsets, top_k=DENSE_TOP_K):
    q_vec = model.encode("query: " + query, normalize_embeddings=True).reshape(1, -1)
    h_vec = model.encode("passage: " + hyde_text, normalize_embeddings=True).reshape(1, -1)
    combined = np.vstack([q_vec, h_vec])
    return _dense_search_vectors(combined, index, payloads, offsets, top_k)


def rrf_fusion(bm25_results, dense_results, top_k=RRF_TOP_K):
    bm25_ranks = {}
    dense_ranks = {}

    for rank, result in enumerate(bm25_results):
        article = result["relevant_article"]
        prev = bm25_ranks.get(article, 999)
        bm25_ranks[article] = min(prev, rank + 1)

    for rank, result in enumerate(dense_results):
        article = result["relevant_article"]
        prev = dense_ranks.get(article, 999)
        dense_ranks[article] = min(prev, rank + 1)

    all_articles = set(bm25_ranks) | set(dense_ranks)

    scored = []
    for article in all_articles:
        bm25_score = 1.0 / (RRF_CONST + bm25_ranks.get(article, 999))
        dense_score = 1.0 / (RRF_CONST + dense_ranks.get(article, 999))
        scored.append((article, bm25_score + dense_score))

    scored.sort(key=lambda x: x[1], reverse=True)

    dense_by_article = {r["relevant_article"]: r for r in dense_results}
    bm25_by_article = {r["relevant_article"]: r for r in bm25_results}

    fused = []
    for article, rrf_score in scored[:top_k]:
        best = dense_by_article.get(article) or bm25_by_article.get(article)
        if not best:
            continue
        entry = dict(best)
        entry.update({
            "rrf_score": rrf_score,
            "bm25_rank": bm25_ranks.get(article),
            "dense_rank": dense_ranks.get(article),
            "relevant_article": article,
        })
        fused.append(entry)

    return fused


def rerank(query, candidates, reranker_model, top_k=FINAL_TOP_K):
    tokenizer, model = reranker_model
    batch_size = 16
    all_scores = []

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        pairs = [(query, c["text"]) for c in batch]
        inputs = tokenizer(
            pairs, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            raw = model(**inputs).logits.squeeze(-1)
        if raw.dim() == 0:
            all_scores.append(raw.item())
        else:
            all_scores.extend(raw.tolist())

    for i, candidate in enumerate(candidates):
        candidate["rerank_score"] = float(all_scores[i])
    candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
    return candidates[:top_k]
