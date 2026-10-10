from qdrant_client import QdrantClient
from qdrant_client.models import Distance,PointStruct, VectorParams
from config import COLLECTION_NAME, QDRANT_HOST,QDRANT_PORT

_client= None   

def _get_client():
    global _client
    if _client is None:
        _client =QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT)

    return _client

def create_collection(vector_size,collection_name=COLLECTION_NAME, recreate=False):
    client =_get_client()
    if recreate and client.collection_exists(collection_name) :
        client.delete_collection(collection_name)
    if not client.collection_exists(collection_name):
        client.create_collection(collection_name=collection_name, vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE))

def upsert_chunks(chunks, vectors, collection_name=COLLECTION_NAME):
    """
    chunks: list[dict] with keys chunk_id (int), source (str), text (str)
    vectors: list[list[float]] — same length as chunks
    """
    client =_get_client()
    points=[PointStruct(id= chunk["chunk_id"], vector=vector, payload={"source": chunk["source"], "text": chunk["text"]},
                        )
                        for chunk, vector in zip(chunks, vectors)
            ]
    client.upsert(collection_name=collection_name, points=points)

def search(query_vector, top_k =5,collection_name=COLLECTION_NAME):
    client =_get_client()
    response =client.query_points(collection_name=collection_name, query=query_vector, limit=top_k)
    hits =response.points
    
    return [
        {
            "chunk_id": hit.id,
            "score": hit.score,
            "source": hit.payload.get("source"),
            "text": hit.payload.get("text"),
        }

        for hit in hits
    ]