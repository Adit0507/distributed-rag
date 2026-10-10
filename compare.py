"""
Runs a list of queries against both BM25 and semantic search,
and logs the results side by side for comparison.
"""

import json

from bm25_index import BM25Index
from config import BM25_INDEX_PATH, QUERIES_PATH, RESULTS_PATH, TOP_K
from embedder import embed
from vector_store import search as semantic_search


def load_queries(path=QUERIES_PATH):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def run_comparison(queries, top_k=TOP_K):
    bm25 = BM25Index.load(BM25_INDEX_PATH)
    results = []

    for query in queries:
        bm25_hits = bm25.search(query, top_k=top_k)
        query_vector = embed([query])[0]
        semantic_hits = semantic_search(query_vector, top_k=top_k)

        results.append(
            {
                "query": query,
                "bm25_top_k": [
                    {"chunk_id": h["chunk_id"], "source": h["source"], "score": h["score"]}
                    for h in bm25_hits
                ],
                "semantic_top_k": [
                    {"chunk_id": h["chunk_id"], "source": h["source"], "score": h["score"]}
                    for h in semantic_hits
                ],
                # Quick eyeball metric: how much do the two methods agree on top-k?
                "overlap_count": len(
                    {h["chunk_id"] for h in bm25_hits} & {h["chunk_id"] for h in semantic_hits}
                ),
            }
        )
    return results


def main():
    queries = load_queries()
    print(f"Running {len(queries)} queries through both search methods...")
    results = run_comparison(queries)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    avg_overlap = sum(r["overlap_count"] for r in results) / len(results)
    print(f"Done. Results written to {RESULTS_PATH}")
    print(f"Average top-{TOP_K} overlap between BM25 and semantic: {avg_overlap:.1f}/{TOP_K}")


if __name__ == "__main__":
    main()
