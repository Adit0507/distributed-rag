import pickle
import re

from rank_bm25 import BM25Okapi
from config import BM25_INDEX_PATH

_TOKEN_RE = re.compile(r"[a-z0-9]+")

def tokenize(text):
    return _TOKEN_RE.findall(text.lower())

class BM25Index:
    def __init__(self, chunks):
        self.chunks =chunks
        tokenized_corpus =[tokenize(chunk["text"]) for chunk in chunks]
        self._bm25 =BM25Okapi(tokenized_corpus)

    def search(self, query,top_k=5):
        tokenized_query =tokenize(query)
        scores =self._bm25.get_scores(tokenized_query) 
        ranked =sorted(range(len(scores)),key=lambda i: scores[i], reverse=True)[:top_k]

        return [
             {
                "chunk_id": self.chunks[i]["chunk_id"],
                "score": float(scores[i]),
                "source": self.chunks[i]["source"],
                "text": self.chunks[i]["text"],
            }
            for i in ranked
        ]
    def save(self, path=BM25_INDEX_PATH):
        with open(path,"wb") as f:
            pickle.dump(self,f)

    @staticmethod
    def load(path=BM25_INDEX_PATH):
        with open(path, "rb") as f:
            return pickle.load(f)