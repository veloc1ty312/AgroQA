import os
from flask import Flask, request, jsonify
from retriever import Retriever
from models import answer
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
retr = Retriever()

@app.get("/health")
def health():
    return jsonify({"ok": True})

@app.post("/chat")
def chat():
    data = request.get_json(force=True, silent=True) or {}
    q = data.get("q", "").strip()
    if not q:
        return jsonify({"error": "Missing 'q'"}), 400

    mode = data.get("mode", "short")
    filters = data.get("filters")  # e.g., {"source": "sprayer_manual.pdf"}
    k = data.get("k", 5)

    docs = retr.search(q, k=k, filters=filters)
    out = answer(q, docs, mode=mode)
    # return source mapping for UI to show clickable citations
    citations = [
        {"idx": i + 1, "source": d["meta"].get("source"), "page": d["meta"].get("page"), "score": d.get("score")}
        for i, d in enumerate(docs)
    ]
    return jsonify({"answer": out, "citations": citations})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)