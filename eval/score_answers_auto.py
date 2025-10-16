import json, re, statistics
from pathlib import Path
from datetime import datetime

IN = Path("eval/answers.jsonl")
OUT_TXT = Path("eval/answer_quality_report.txt")
OUT_CSV = Path("eval/answer_quality_auto.csv")
HUMAN_TEMPLATE = Path("eval/answer_quality_human_template.csv")

BRACKET_RE = re.compile(r"\[(\d+)\]")

def sentences(text: str):
    if not text:
        return []
    # simple splitter on . ! ? with some trimming
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if p.strip()]

def load_answers(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8-sig") as f:
        for raw in f:
            s = raw.strip()
            if not s or s.startswith("{\"_run_meta\""):
                continue
            rec = json.loads(s)
            rows.append(rec)
    return rows

def analyze(rec):
    ans = rec.get("answer") or ""
    cits = rec.get("citations") or []
    # derive valid indices
    idxs = []
    if cits and isinstance(cits[0], dict) and "idx" in cits[0]:
        idxs = {int(c["idx"]) for c in cits if isinstance(c.get("idx"), int)}
    else:
        idxs = set(range(1, len(cits)+1))  # fallback 1..len

    sent_list = sentences(ans)
    sent_n = len(sent_list)

    nums = [int(n) for n in BRACKET_RE.findall(ans)]
    unmatched = [n for n in nums if n not in idxs]

    # grounding proxy: fraction of sentences containing at least one [#]
    grounded_sents = 0
    if sent_n:
        for s in sent_list:
            if BRACKET_RE.search(s):
                grounded_sents += 1
    grounded_frac = (grounded_sents / sent_n) if sent_n else 0.0

    short_len_ok = (3 <= sent_n <= 5) if rec.get("mode") == "short" else None

    return {
        "id": rec.get("id"),
        "mode": rec.get("mode"),
        "sentences": sent_n,
        "has_brackets": len(nums) > 0,
        "unmatched_citation_numbers": unmatched,
        "grounded_fraction": grounded_frac,
        "short_len_ok": short_len_ok
    }

def write_human_template(rows):
    # Columns for human scoring: 1–5 scale + optional notes
    with open(HUMAN_TEMPLATE, "w", encoding="utf-8") as f:
        f.write("id,mode,Grounding_1to5,Correctness_1to5,Actionability_1to5,CitationFidelity_1to5,Notes\n")
        for r in rows:
            f.write(f"{r['id']},{r['mode']},,,,,""\n")

def main():
    rows = load_answers(IN)
    if not rows:
        print(f"No rows in {IN}. Run collect_answers.py first.")
        return

    per = [analyze(r) for r in rows]

    # aggregate by mode
    modes = {}
    for p in per:
        m = p["mode"]
        modes.setdefault(m, []).append(p)

    OUT_TXT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write("Answer Quality (automatic checks)\n")
        f.write(f"Run: {datetime.now().isoformat(timespec='seconds')}\n")
        f.write(f"Source: {IN}\n\n")

        for m, items in modes.items():
            sent_counts = [x["sentences"] for x in items]
            grounded = [x["grounded_fraction"] for x in items]
            has_br = sum(1 for x in items if x["has_brackets"])
            unmatched_any = sum(1 for x in items if x["unmatched_citation_numbers"])
            f.write(f"Mode: {m}\n")
            f.write(f"  N: {len(items)}\n")
            f.write(f"  Avg sentences: {statistics.mean(sent_counts):.2f}\n")
            f.write(f"  Median sentences: {statistics.median(sent_counts):.0f}\n")
            f.write(f"  Answers with any [#]: {has_br}/{len(items)}\n")
            f.write(f"  Answers with unmatched [#]: {unmatched_any}/{len(items)}\n")
            f.write(f"  Avg grounded sentence fraction: {statistics.mean(grounded):.2f}\n")
            if m == "short":
                short_ok = [x["short_len_ok"] for x in items if x["short_len_ok"] is not None]
                if short_ok:
                    f.write(f"  Short length in 3–5 sentences: {sum(1 for v in short_ok if v)}/{len(short_ok)}\n")
            f.write("\n")

    # detailed CSV per answer
    with open(OUT_CSV, "w", encoding="utf-8") as f:
        f.write("id,mode,sentences,has_brackets,unmatched_count,grounded_fraction,short_len_ok\n")
        for x in per:
            f.write(f"{x['id']},{x['mode']},{x['sentences']},{int(x['has_brackets'])},{len(x['unmatched_citation_numbers'])},{x['grounded_fraction']:.2f},{'' if x['short_len_ok'] is None else int(x['short_len_ok'])}\n")

    write_human_template(rows)

    print(f"Wrote {OUT_TXT}")
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {HUMAN_TEMPLATE} (fill 1–5 rubric by hand, then you can average per mode)")

if __name__ == "__main__":
    main()