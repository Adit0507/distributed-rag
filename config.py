"""Configuration for rag engine """

EMBEDDING_MODEL_NAME="all-MiniLM-L6-v2"

QDRANT_HOST ="localhost"
QDRANT_PORT = 6333
COLLECTION_NAME = "distributed_systems_corpus"

CHUNK_SIZE_WORDS= 400
CHUNK_OVERLAP_WORDS = 0

CORPUS_DIR ="corpus"
BM25_INDEX_PATH = "bm25_index.pkl"
QUERIES_PATH = "queries.txt"
RESULTS_PATH = "comparison_results.json"

# search
TOP_K =5 