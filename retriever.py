import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_DIR = "indexes/chroma"
COLLECTION_NAME = "agroqa"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

def _build_where(filters: dict | None):
    if not filters:
        return None
    if any(isinstance(k, str) and k.startswith("$") for k in filters.keys()):
        return filters
    clauses = [{k: {"$eq": v}} for k, v in filters.items()]
    return {"$and": clauses}

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
        where = _build_where(filters)

        res = self.col.query(
            query_texts=[query],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        hits = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        dists = res.get("distances", [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            score = 1.0 / (1.0 + float(dist)) if dist is not None else None
            hits.append({"text": text, "meta": meta, "score": score})
        return hits