from config import EMBEDDING_MODEL_NAME
from sentence_transformers import SentenceTransformer

_model =None

def _get_model():
    global _model
    if _model is None:
        _model =SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model

def embed(texts):
    """Embed a list of string into vectors """
    model = _get_model()
    vectors =model.encode(texts, show_progress_bar=False,convert_to_numpy=True)
    return vectors.tolist()

def embedding_dim():
    model =_get_model()
    return model.get_embedding_dimension()

if __name__== "__main__":
    import numpy as np
    samples = [
        "Raft is a consensus algorithm for replicated logs.",
        "Raft elects a leader to manage log replication across a cluster.",
        "The recipe calls for two cups of flour and a pinch of salt.",
    ]
    vecs =np.array(embed(samples))
    def cosine(a, b):
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

    print(f"Embedding dimension: {embedding_dim()}")
    print(f"sim(raft, raft-leader) = {cosine(vecs[0], vecs[1]):.4f}  (expect high)")
    print(f"sim(raft, recipe)      = {cosine(vecs[0], vecs[2]):.4f}  (expect low)")