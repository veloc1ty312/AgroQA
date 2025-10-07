import os
import uuid
import fitz
from typing import Iterator, Tuple
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

SRC_DIR = "data/raw"
DB_DIR = "indexes/chroma"
COLLECTION_NAME = "agroqa"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def iter_pdf_pages(path: str) -> Iterator[Tuple[int, str]]:
    doc = fitz.open(path)
    for i, page in enumerate(doc):
        yield i + 1, page.get_text("text")

def chunk_text(text: str, size: int = 1200, overlap: int = 200):
    i = 0
    n = len(text)
    while i < n:
        yield text[i : i + size]
        i += max(1, size - overlap)

def main():
    os.makedirs(DB_DIR, exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR)
    emb_fn = SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
    col = client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=emb_fn)

    total = 0
    docs, ids, metas = [], [], []

    for name in os.listdir(SRC_DIR):
        if not name.lower().endswith(".pdf"):
            continue
        src_path = os.path.join(SRC_DIR, name)
        for page_num, text in iter_pdf_pages(src_path):
            if not text or not text.strip():
                continue
            for c in chunk_text(text):
                c = c.strip()
                if not c:
                    continue
                ids.append(str(uuid.uuid4()))
                docs.append(c)
                metas.append({"source": name, "page": page_num})
                total += 1

                # batch insert every 1k to keep memory steady
                if len(ids) >= 1000:
                    col.add(documents=docs, metadatas=metas, ids=ids)
                    docs, metas, ids = [], [], []

    if ids:
        col.add(documents=docs, metadatas=metas, ids=ids)

    print(f"Ingested {total} chunks into Chroma at {DB_DIR} (collection='{COLLECTION_NAME}').")

if __name__ == "__main__":
    main()