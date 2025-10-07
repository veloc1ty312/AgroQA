import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_DIR = "indexes/chroma"
COLLECTION_NAME = "agroqa"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

class Retriever:
    def __init__(self, k: int = 5):
        self.client = chromadb.PersistentClient(path=DB_DIR)
        self.col = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=SentenceTransformerEmbeddingFunction(model_name=EMB_MODEL)
        )
        self.k = k

    def search(self, query: str, k: int | None = None, filters: dict | None = None):
        k = k or self.k
        res = self.col.query(
            query_texts=[query],
            n_results=k,
            where=filters or {},
            include=["documents", "metadatas", "distances", "ids"]
        )
        hits = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            score = 1.0 / (1.0 + float(dist)) if dist is not None else None
            hits.append({"text": text, "meta": meta, "score": score})
        return hits