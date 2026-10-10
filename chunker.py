""" loads raw docs from disk and splits them into fixed size, non overlapping chunks """
from pathlib import Path
from config import CHUNK_SIZE_WORDS, CORPUS_DIR

SUPPORTED_EXTENSIONS = {".txt", ".md"}

def load_documents(corpus_dir=CORPUS_DIR):
    docs =[]
    corpus_path =Path(corpus_dir)
    if not corpus_path.exists(): 
        raise FileNotFoundError(
            f"Corpus directory '{corpus_dir}' not found. Create it and drop your "
            f".txt/.md source documents inside."
        )
    for file_path in sorted(corpus_path.iterdir()):
        if file_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        text = file_path.read_text(encoding="utf-8", errors="ignore")
        docs.append({"source": file_path.name, "text": text})

    if not docs:
        raise ValueError(
            f"No .txt/.md files found in '{corpus_dir}'. Add your distributed "
            f"systems papers and blog posts there first."
        )
    return docs

def chunk_documents(docs, chunk_size_words=CHUNK_SIZE_WORDS):
    chunks= []
    chunk_id =0
    for doc in docs:
        words= doc["text"].split()
        for i in range(0, len(words), chunk_size_words):
            chunk_words =words[i: i+chunk_size_words]
            if not chunk_words:
                continue
            chunks.append({
                "chunk_id": chunk_id,
                "source": doc["source"],
                "text": " ".join(chunk_words),
            })
            chunk_id +=1
    return chunks

if __name__== "__main__":
    docs= load_documents()
    chunks =chunk_documents(docs)
    print(f"Loaded {len(docs)} documents -> {len(chunks)} chunks")
    print(f"Example chunk: {chunks[0]['source']} chunk_id={chunks[0]['chunk_id']}")
    print(chunks[0]["text"][:200] + "...")