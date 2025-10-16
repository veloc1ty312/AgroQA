import os
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

DB_DIR = "indexes/chroma"
COLLECTION_NAME = "agroqa"
EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RAW_DIR = "data/raw"

def _expand_contains_clause(k, v):
    if isinstance(v, dict) and "$contains" in v and k == "source":
        sub = str(v["$contains"]).lower()
        try:
            names = [fn for fn in os.listdir(RAW_DIR) if fn.lower().endswith(".pdf") and sub in fn.lower()]
        except Exception:
            names = []
        if names:
            return {k: {"$in": names}}
        else:
            return {k: {"$eq": "__no_match__"}}
    return {k: v if isinstance(v, dict) else {"$eq": v}}

def _normalize_where_dict(w):
    if not isinstance(w, dict):
        return w
    if "$and" in w and isinstance(w["$and"], list):
        terms = [_normalize_where_dict(t) for t in w["$and"]]
        if len(terms) == 1:
            return terms[0]
        return {"$and": terms}
    if "$or" in w and isinstance(w["$or"], list):
        terms = [_normalize_where_dict(t) for t in w["$or"]]
        if len(terms) == 1:
            return terms[0]
        return {"$or": terms}
    out = {}
    for k, v in w.items():
        if isinstance(k, str) and k.startswith("$"):
            out[k] = v
        else:
            out.update(_expand_contains_clause(k, v))
    return out

def _build_where(filters: dict | None):
    if not filters:
        return None
    if "where" in filters and isinstance(filters["where"], dict):
        return _normalize_where_dict(filters["where"])
    if any(isinstance(k, str) and k.startswith("$") for k in filters.keys()):
        return _normalize_where_dict(filters)
    clauses = []
    for k, v in filters.items():
        clauses.append(_expand_contains_clause(k, v))
    if len(clauses) == 1:
        return clauses[0]
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