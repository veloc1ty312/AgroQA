import os
import json
import requests
from pathlib import Path
from datetime import datetime

API_URL = "http://localhost:8000/chat"
JSONL_PATH = Path("eval/seed_qas.jsonl")
OUT_PATH = Path("eval/smoke_results.txt")

def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    if not JSONL_PATH.exists():
        print(f"File not found: {JSONL_PATH.resolve()}")
        return

    total = ok = 0

    with open(OUT_PATH, "w", encoding="utf-8") as out, open(JSONL_PATH, "r", encoding="utf-8-sig") as f:
        out.write(f"AgroQA Smoke Evaluation\n")
        out.write(f"Run at: {datetime.now().isoformat(timespec='seconds')}\n")
        out.write(f"API: {API_URL}\n")
        out.write(f"Seed file: {JSONL_PATH.resolve()}\n")
        out.write("=" * 80 + "\n")

        for ln, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("//"):
                continue

            try:
                ex = json.loads(line)
            except json.JSONDecodeError as e:
                out.write(f"\n[skip] line {ln}: {e.msg} (col {e.colno})\n")
                continue

            q = ex.get("q", "").strip()
            # Keep the original behavior (force short), but include filters if provided
            payload = {"q": q, "mode": "short"}
            if ex.get("filters") is not None:
                payload["filters"] = ex["filters"]

            try:
                r = requests.post(API_URL, json=payload, timeout=60)
            except Exception as e:
                out.write(f"\n[{ln}] Q: {q}\nError: request failed :: {e}\n")
                total += 1
                continue

            out.write(f"\n[{ln}] Q: {q}\n")
            if r.ok:
                js = r.json()
                ans = js.get("answer", "")
                cits = js.get("citations", [])
                out.write("A: " + ans + "\n")
                out.write("Citations: " + json.dumps(cits, ensure_ascii=False) + "\n")
                ok += 1
            else:
                out.write(f"Error: {r.status_code} {r.text}\n")

            total += 1

        out.write("\n" + "=" * 80 + "\n")
        out.write(f"Summary: OK {ok}/{total} ({(ok/total*100 if total else 0):.1f}%)\n")

    print(f"Wrote results to: {OUT_PATH.resolve()}")
    print(f"Summary: OK {ok}/{total} ({(ok/total*100 if total else 0):.1f}%)")

if __name__ == "__main__":
    main()