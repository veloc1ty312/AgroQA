import os
from flask import Flask, request, jsonify
from retriever import Retriever
from models import answer
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__, static_folder="ui", static_url_path="/ui")
retr = Retriever()

@app.get("/")
def root():
    return app.send_static_file("index.html")

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
    filters = data.get("filters") or None
    if isinstance(filters, dict) and len(filters) == 0:
        filters = None
    k = data.get("k", 5)

    docs = retr.search(q, k=k, filters=filters)
    out, graph = answer(q, docs, mode=mode)
    citations = [
        {"idx": i + 1, "source": d["meta"].get("source"), "page": d["meta"].get("page"), "score": d.get("score")}
        for i, d in enumerate(docs)
    ]
    return jsonify({"answer": out, "graph": graph, "citations": citations})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)