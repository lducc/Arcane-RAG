import json
import sys
import time
from pathlib import Path

from scripts.retrieval import (
    FINAL_TOP_K,
    load_model,
    load_index,
    load_reranker,
    get_database,
    search_bm25,
    search_dense,
    search_dense_hyde,
    rrf_fusion,
    rerank,
)
from scripts.llm_expander import expand_query, generate_hyde

QUESTIONS_FILE = Path("R2AIStage1DATA (2).json")
OUTPUT_FILE = Path("data/results.json")


def main():
    use_reranker = "--rerank" in sys.argv
    bm25_only = "--bm25-only" in sys.argv
    use_expand = "--expand" in sys.argv
    use_hyde = "--hyde" in sys.argv

    if use_expand or use_hyde:
        from scripts.llm_expander import load_llm
        print("Loading LLM...", file=sys.stderr)
        t0 = time.time()
        load_llm()
        print(f"  LLM loaded in {time.time() - t0:.0f}s", file=sys.stderr)

    questions = json.loads(QUESTIONS_FILE.read_text("utf-8"))
    print(f"Loaded {len(questions)} questions", file=sys.stderr)

    print("Loading models...", file=sys.stderr)
    load_start = time.time()

    dense_model = None
    index = None
    payloads = None
    offsets = None
    reranker_model = None

    if not bm25_only:
        dense_model = load_model()
        index, payloads, offsets = load_index()
        if use_reranker:
            reranker_model = load_reranker()

    print(f"  Loaded in {time.time() - load_start:.1f}s", file=sys.stderr)

    results = []
    total_time = 0.0

    for i, question_data in enumerate(questions):
        query_start = time.time()
        query = question_data["question"]

        connection = get_database()
        bm25_hits = search_bm25(connection, query)
        connection.close()

        fused = bm25_hits[:FINAL_TOP_K]
        dense_hits = []

        if dense_model and index:
            if use_expand:
                expanded = expand_query(query)
                all_queries = [query] + expanded
                bm25_hits = []
                dense_hits = []
                for q in all_queries:
                    conn = get_database()
                    bm25_hits.extend(search_bm25(conn, q))
                    conn.close()
                    dense_hits.extend(search_dense(q, dense_model, index, payloads, offsets))
            elif use_hyde:
                dense_hits = search_dense(query, dense_model, index, payloads, offsets)
                hyde_text = generate_hyde(query)
                dense_hits += search_dense_hyde(query, hyde_text, dense_model, index, payloads, offsets)
            else:
                dense_hits = search_dense(query, dense_model, index, payloads, offsets)

            fused = rrf_fusion(bm25_hits, dense_hits)

        if reranker_model and fused:
            fused = rerank(query, fused, reranker_model)

        top_articles = [result["relevant_article"] for result in fused[:FINAL_TOP_K]]

        elapsed = time.time() - query_start
        total_time += elapsed

        results.append({
            "id": question_data["id"],
            "question": query,
            "predicted_articles": top_articles,
            "top_article": top_articles[0] if top_articles else "",
            "bm25_hits": len(bm25_hits),
            "dense_hits": len(dense_hits),
            "fused": len(fused),
            "time": round(elapsed, 3),
        })

        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(questions)}] avg query: {total_time / (i + 1):.2f}s", file=sys.stderr)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, ensure_ascii=False))

    with_results = sum(1 for r in results if r["top_article"])
    average_time = total_time / len(questions)
    print(f"\nDone — {len(results)} queries processed", file=sys.stderr)
    print(f"Average query time: {average_time:.2f}s", file=sys.stderr)
    print(f"Queries with results: {with_results}/{len(results)}", file=sys.stderr)
    print(f"Results saved to {OUTPUT_FILE}", file=sys.stderr)


if __name__ == "__main__":
    main()
