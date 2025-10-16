import json, requests, time
from pathlib import Path
from datetime import datetime

API = "http://localhost:8000/chat"
GOLD = Path("eval/gold_labels.jsonl")
OUT = Path("eval/answers.jsonl")

K = 5
MODES = ["short", "long"]   # collect both to compare personalization

def iter_gold(path: Path):
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, raw in enumerate(f, 1):
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            ex = json.loads(s)
            yield {
                "id": ex.get("id", f"q{ln}"),
                "q": ex["q"].strip(),
                "filters": ex.get("filters")  # may be None
            }

def ask(q, mode="short", k=K, filters=None, timeout=90):
    payload = {"q": q, "k": k, "mode": mode}
    if filters is not None:
        payload["filters"] = filters
    r = requests.post(API, json=payload, timeout=timeout)
    if not r.ok:
        return None, f"http_{r.status_code}", r.text[:200]
    try:
        js = r.json()
    except Exception as e:
        return None, "json_error", str(e)
    return js, "ok", ""

def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(OUT, "w", encoding="utf-8") as out:
        out.write(json.dumps({"_run_meta": {
            "api": API, "gold": str(GOLD), "k": K,
            "timestamp": datetime.now().isoformat(timespec="seconds")
        }}) + "\n")

        for item in iter_gold(GOLD):
            for mode in MODES:
                js, status, err = ask(item["q"], mode=mode, k=K, filters=None)  # unfiltered pass
                rec = {
                    "id": item["id"],
                    "q": item["q"],
                    "mode": mode,
                    "filters": None,
                    "status": status,
                    "error": err,
                    "answer": (js or {}).get("answer"),
                    "citations": (js or {}).get("citations", [])
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n += 1
                time.sleep(0.1)

    print(f"Wrote {OUT} with {n} rows (both modes, unfiltered).")

if __name__ == "__main__":
    main()