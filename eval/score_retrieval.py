import json, requests, math
from pathlib import Path

API = "http://localhost:8000/chat"
GOLD = Path("eval/gold_labels.jsonl")
OUT_TXT = Path("eval/retrieval_report.txt")

K = 5  # top-k

def load_gold(path: Path):
    items = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for ln, raw in enumerate(f, 1):
            s = raw.strip()
            if not s or s.startswith("#") or s.startswith("//"):
                continue
            ex = json.loads(s)
            q = ex["q"].strip()
            srcs = [t.lower() for t in ex.get("expect_sources", [])]
            pages = ex.get("expect_pages", [])
            flt = ex.get("filters", None)
            items.append({"q": q, "expect_sources": srcs, "expect_pages": pages, "filters": flt, "ln": ln})
    return items

def within_pages(page, ranges):
    if not ranges:
        return True
    for r in ranges:
        if isinstance(r, int):
            if page == r:
                return True
        elif isinstance(r, (list, tuple)) and len(r) == 2:
            lo, hi = r
            if lo <= page <= hi:
                return True
    return False

def match(citation, expect_sources, expect_pages):
    src = (citation.get("source") or "").lower()
    pg = citation.get("page")
    if not expect_sources:
        return False
    if any(t in src for t in expect_sources):
        if pg is None:
            return True if not expect_pages else False
        return within_pages(pg, expect_pages)
    return False

def ask(q, k=K, mode="short", filters=None):
    payload = {"q": q, "k": k, "mode": mode}
    if filters is not None:
        payload["filters"] = filters
    try:
        r = requests.post(API, json=payload, timeout=90)
    except Exception:
        return None
    if not r.ok:
        return None
    return r.json()

def evaluate(items, use_filters=False):
    hits = 0
    rr_sum = 0.0
    n = 0

    for it in items:
        q = it["q"]
        expect_sources = it["expect_sources"]
        expect_pages = it["expect_pages"]
        filters = it["filters"] if use_filters else None

        js = ask(q, k=K, mode="short", filters=filters)
        if not js or "citations" not in js:
            continue

        cits = js["citations"][:K]
        n += 1
        found_rank = None
        for i, c in enumerate(cits, start=1):
            if match(c, expect_sources, expect_pages):
                found_rank = i
                break

        if found_rank is not None:
            hits += 1
            rr_sum += 1.0 / found_rank

    recall = hits / n if n else 0.0
    mrr = rr_sum / n if n else 0.0
    return recall, mrr, n

def write_outputs(recall_unf, mrr_unf, n_unf, recall_filt, mrr_filt, n_filt):
    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Retrieval Evaluation\n")
        f.write(f"API: {API}\n")
        f.write(f"K: {K}\n\n")
        f.write("== Unfiltered ==\n")
        f.write(f"Questions: {n_unf}\nRecall@{K}: {recall_unf:.3f}\nMRR@{K}: {mrr_unf:.3f}\n\n")
        f.write("== With Filters ==\n")
        f.write(f"Questions: {n_filt}\nRecall@{K}: {recall_filt:.3f}\nMRR@{K}: {mrr_filt:.3f}\n")

def main():
    items = load_gold(GOLD)
    if not items:
        print(f"No items loaded from {GOLD}")
        return

    recall_unf, mrr_unf, n_unf = evaluate(items, use_filters=False)
    recall_filt, mrr_filt, n_filt = evaluate(items, use_filters=True)

    write_outputs(recall_unf, mrr_unf, n_unf, recall_filt, mrr_filt, n_filt)
    print(f"Wrote {OUT_TXT}")
    print(f"Unfiltered: Recall@{K}={recall_unf:.3f}, MRR@{K}={mrr_unf:.3f} over {n_unf} Qs")
    print(f"Filtered: Recall@{K}={recall_filt:.3f}, MRR@{K}={mrr_filt:.3f} over {n_filt} Qs")

if __name__ == "__main__":
    main()