# running to populate qdrant and build bm25 index

from bm25_index import BM25Index
from chunker import chunk_documents, load_documents
from config import BM25_INDEX_PATH
from embedder import embed, embedding_dim
from vector_store import create_collection,upsert_chunks

def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"  {len(documents)} documents loaded")

    print("Chunking....")
    chunks =chunk_documents(documents)
    print(f"  {len(chunks)} chunks created")

    print("Embedding chunks (slow step, runs on CPU)...")
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed(texts)

    print("Pushing to Qdrant...")
    create_collection(vector_size=embedding_dim(), recreate=True)
    upsert_chunks(chunks, vectors)
    print(f"  {len(chunks)} vectors upserted")

    print("Building BM25 index...")
    bm25 = BM25Index(chunks)
    bm25.save(BM25_INDEX_PATH)
    print(f"  BM25 index saved to {BM25_INDEX_PATH}")

    print("\nIngestion complete. Run compare.py next.")

if __name__== "__main__":
    main()