import os
import re
from flask import Flask, request, jsonify
from retriever import Retriever
from models import answer
from dotenv import load_dotenv
import io, base64, ast
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def _safe_exec_matplotlib(code: str) -> str:
    tree = ast.parse(code, mode="exec")
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal,
                             ast.With, ast.Try, ast.Raise, ast.Delete, ast.ClassDef,
                             ast.AsyncFunctionDef, ast.Lambda)):
            raise ValueError("Disallowed Python construct.")
        if isinstance(node, ast.Attribute):
            if isinstance(node.attr, str) and node.attr.startswith("__"):
                raise ValueError("Disallowed attribute access.")
    safe_globals = {
        "__builtins__": {"range": range, "len": len, "min": min, "max": max, "sum": sum, "abs": abs},
        "plt": plt, "np": np,
    }
    safe_locals = {}
    plt.close("all")
    exec(compile(tree, "<graph>", "exec"), safe_globals, safe_locals)
    fig = plt.gcf()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    plt.close(fig)
    return "data:image/png;base64," + b64

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
    
    out = re.sub(r"```[\s\S]*?```", "", out).strip()
    lines = []
    for line in out.splitlines():
        if re.search(r"(?i)\b(matplotlib|seaborn|plotly|plt\.)", line):
            continue
        if re.search(r"(?i)\b(chart|graph|plot|figure)\b", line):
            continue
        lines.append(line)
    out = "\n".join(lines).strip()

    if graph and graph.strip().upper() != "N/A":
        try:
            graph = _safe_exec_matplotlib(graph)
        except Exception as e:
            graph = None
    
    citations = [
        {"idx": i + 1, "source": d["meta"].get("source"), "page": d["meta"].get("page"), "score": d.get("score")}
        for i, d in enumerate(docs)
    ]
    return jsonify({"answer": out, "graph_image": graph, "citations": citations})

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    app.run(host="0.0.0.0", port=port, debug=True)