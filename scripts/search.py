import sys
import time

from scripts.retrieval import (
    INDEX_DIR,
    FINAL_TOP_K,
    RRF_TOP_K,
    load_model,
    load_index,
    load_reranker,
    get_database,
    search_bm25,
    search_dense,
    rrf_fusion,
    rerank,
)


def format_result(result, index):
    lines = [f"--- Result {index} ---"]

    lines.append(f"Article: {result.get('relevant_article', 'N/A')}")
    lines.append(f"Doc: {result.get('doc_name', 'N/A')}")

    if result.get("rrf_score"):
        lines.append(f"RRF Score: {result['rrf_score']:.4f}")
    if result.get("rerank_score"):
        lines.append(f"Re-rank Score: {result['rerank_score']:.4f}")
    if result.get("bm25_rank"):
        lines.append(f"BM25 Rank: {result['bm25_rank']}")
    if result.get("dense_rank"):
        lines.append(f"Dense Rank: {result['dense_rank']}")

    lines.append(f"Chunk: {result.get('chunk_id', 'N/A')}")
    lines.append(f"URL: {result.get('source_url', 'N/A')}")
    lines.append(f"Text: {result.get('text', '')[:300]}...")

    return "\n".join(lines)


def main():
    use_reranker = "--rerank" in sys.argv
    bm25_only = not (INDEX_DIR / "index.faiss").exists()

    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/search.py <query> [top_k] [--rerank]")
        sys.exit(1)

    query = sys.argv[1]
    top_k = FINAL_TOP_K
    if len(sys.argv) > 2 and sys.argv[2] != "--rerank":
        top_k = int(sys.argv[2])

    print(f"Query: {query}", file=sys.stderr)
    start = time.time()

    connection = get_database()
    bm25_results = search_bm25(connection, query)
    connection.close()
    print(f"  BM25 hits: {len(bm25_results)} ({time.time() - start:.1f}s)", file=sys.stderr)

    fused = []

    if not bm25_only:
        print("Dense search...", file=sys.stderr)
        model = load_model()
        index, payloads, offsets = load_index()
        dense_results = search_dense(query, model, index, payloads, offsets)
        print(f"  Dense hits: {len(dense_results)} ({time.time() - start:.1f}s)", file=sys.stderr)

        if bm25_results or dense_results:
            print("RRF fusion...", file=sys.stderr)
            fused = rrf_fusion(bm25_results, dense_results, top_k=RRF_TOP_K)
            print(f"  Fused articles: {len(fused)} ({time.time() - start:.1f}s)", file=sys.stderr)

    if not fused:
        fused = bm25_results[:top_k]

    if not fused:
        print("No results found.")
        return

    if use_reranker and fused:
        print("Re-ranking...", file=sys.stderr)
        reranker = load_reranker()
        fused = rerank(query, fused, reranker, top_k=top_k)
        print(f"  Re-ranked: {len(fused)} ({time.time() - start:.1f}s)", file=sys.stderr)

    print(file=sys.stderr)
    for i, result in enumerate(fused[:top_k], 1):
        print(format_result(result, i))
        print()


if __name__ == "__main__":
    main()
