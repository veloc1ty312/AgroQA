import os
import re
import time
import json
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Tuple, Optional, Iterable

import requests
from bs4 import BeautifulSoup

SEEDS_FILE = "seeds.jsonl"
SRC_DIR = os.path.join("data", "raw")
os.makedirs(SRC_DIR, exist_ok=True)

HDRS = {"User-Agent": "AgroQA-pdf-fetcher/0.1 (+contact@example.com)"}
REQUEST_TIMEOUT = 45
SLEEP_BETWEEN_DOWNLOADS = 0.5

# keyword gating
POS = re.compile(
    r"(?i)\b("
    r"irrigation|evapotranspiration|ET|soil moisture|soil-water|water management|"
    r"maize|corn|center[-\s]?pivot|sprinkler|drip|micro[-\s]?irrigation|"
    r"fertigation|nozzle|pressure regulator|sprayer|tractor|pump|flowmeter|"
    r"irrigation scheduling|water use|crop water|deficit irrigation|pivot"
    r")\b"
)

NEG = re.compile(
    r"(?i)\b("
    r"civil[_\-\s]?rights|harassment|privacy|about|careers|jobs|press|news|media|"
    r"accessibility|terms|copyright|cookies|donate|alumni|events|campus|library|"
    r"equity|affirmative|nda|policy|compliance|grant|award|newsletter"
    r")\b"
)

def canon(u: str) -> str:
    p = urllib.parse.urlsplit(u)
    scheme = "https" if p.scheme in ("http", "https") else p.scheme
    netloc = p.netloc.lower()
    path = re.sub(r"/+", "/", p.path).rstrip("/")
    return urllib.parse.urlunsplit((scheme, netloc, path, p.query, ""))

def out_path_for(url: str, base_dir: str) -> str:
    name = urllib.parse.quote_plus(url)
    ext = os.path.splitext(urllib.parse.urlsplit(url).path)[1].lower()
    if not ext:
        ext = ".pdf"
    return os.path.join(base_dir, f"{name}{ext}")

def get_sitemap_urls(sitemap_url: str) -> List[str]:
    try:
        r = requests.get(sitemap_url, headers=HDRS, timeout=REQUEST_TIMEOUT)
        r.raise_for_status()
    except Exception as e:
        print("sitemap error:", sitemap_url, e)
        return []

    content = r.content
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        try:
            import gzip
            root = ET.fromstring(gzip.decompress(content))
        except Exception:
            print("sitemap parse error:", sitemap_url)
            return []

    def tag_endswith(elem, name: str) -> bool:
        return elem.tag.endswith(name)

    urls: List[str] = []

    # sitemap index
    if tag_endswith(root, "sitemapindex"):
        for sm in root.findall(".//*"):
            if tag_endswith(sm, "loc") and sm.text:
                child = sm.text.strip()
                urls.extend(get_sitemap_urls(child))
        return urls

    # url set
    if tag_endswith(root, "urlset"):
        for loc in root.findall(".//*"):
            if tag_endswith(loc, "loc") and loc.text:
                urls.append(loc.text.strip())
        return urls

    # fallback: any <loc>
    for loc in root.findall(".//*"):
        if tag_endswith(loc, "loc") and loc.text:
            urls.append(loc.text.strip())
    return urls

def compile_patterns(patterns: Optional[Iterable[str]]) -> List[re.Pattern]:
    if not patterns:
        return []
    return [re.compile(p, flags=re.I) for p in patterns]

def allowed(url: str, allow_patterns: List[re.Pattern], deny_patterns: List[re.Pattern]) -> bool:
    if allow_patterns and not any(p.search(url) for p in allow_patterns):
        return False
    if deny_patterns and any(p.search(url) for p in deny_patterns):
        return False
    return True

def _is_pdf_response(r: requests.Response, url: str) -> bool:
    ct = r.headers.get("Content-Type", "").lower()
    return ("pdf" in ct) or url.lower().endswith(".pdf")

def normalize_text_for_match(s: str) -> str:
    s = urllib.parse.unquote(s)
    s = re.sub(r"[_\-]+", " ", s)
    return s

def topical_score(url: str, title: Optional[str] = None) -> int:
    hay = normalize_text_for_match(url)
    if title:
        hay += " " + normalize_text_for_match(title)
    pos_hits = POS.findall(hay)
    neg_hits = NEG.findall(hay)
    score = len(set(map(str.lower, pos_hits))) - (2 * len(set(map(str.lower, neg_hits))))
    return score

def topical_enough(url: str, title: Optional[str] = None, min_score: int = 1) -> bool:
    return topical_score(url, title) >= min_score

def pick_best_pdf_from_html(base_url: str, html_text: str, min_score: int = 1) -> Optional[str]:
    soup = BeautifulSoup(html_text, "html.parser")
    page_title = soup.title.get_text(strip=True) if soup.title else ""
    candidates = []
    for a in soup.find_all("a", href=True):
        href = urllib.parse.urljoin(base_url, a["href"])
        if not href.lower().endswith(".pdf"):
            continue
        anchor = a.get_text(" ", strip=True) or ""
        score = topical_score(href, f"{anchor} {page_title}".strip())
        candidates.append((score, href))
    if not candidates:
        return None
    candidates.sort(reverse=True, key=lambda t: t[0])
    best_score, best_href = candidates[0]
    if best_score < min_score:
        return None
    return best_href

def download_pdf(url: str) -> Tuple[bool, Optional[str]]:
    # direct hit
    out = out_path_for(url, SRC_DIR)
    if os.path.exists(out):
        return False, out

    try:
        r = requests.get(url, headers=HDRS, timeout=REQUEST_TIMEOUT, allow_redirects=True)
    except Exception as e:
        print("download error:", url, e)
        return False, None

    if r.status_code != 200:
        return False, None

    # direct pdf
    if _is_pdf_response(r, url):
        if not topical_enough(url):
            sc = topical_score(url)
            print(f"skip (nontopical, score={sc}):", url)
            return False, None
        with open(out, "wb") as f:
            f.write(r.content)
        return True, out

    # if html, pick best pdf candidate
    ct = r.headers.get("Content-Type", "").lower()
    if "html" in ct or r.text.lstrip().startswith("<!DOCTYPE"):
        pdf_href = pick_best_pdf_from_html(r.url, r.text, min_score=1)
        if not pdf_href:
            print("skip (no topical pdf on page):", r.url)
            return False, None
        out2 = out_path_for(pdf_href, SRC_DIR)
        if os.path.exists(out2):
            return False, out2
        try:
            rr = requests.get(pdf_href, headers=HDRS, timeout=REQUEST_TIMEOUT)
            if rr.status_code == 200 and _is_pdf_response(rr, pdf_href):
                if not topical_enough(pdf_href):
                    sc2 = topical_score(pdf_href)
                    print(f"skip (nontopical candidate, score={sc2}):", pdf_href)
                    return False, None
                with open(out2, "wb") as f:
                    f.write(rr.content)
                print("picked:", pdf_href, "from", r.url)
                return True, out2
        except Exception as e:
            print("download error:", pdf_href, e)
            return False, None

    return False, None

def process_seed(seed: dict):
    allow = compile_patterns(seed.get("allow") or [])
    deny = compile_patterns(seed.get("deny") or [])
    # accept both "sitemaps" and "sitemap_urls"
    sitemaps = seed.get("sitemaps") or seed.get("sitemap_urls") or []

    if not sitemaps:
        print("skip seed (no sitemaps):", seed.get("domain") or "(unknown domain)")
        return

    seen = set()  # canonical URLs seen across all sitemaps in this seed

    for sm in sitemaps:
        urls = get_sitemap_urls(sm)
        for u in urls:
            cu = canon(u)
            if cu in seen:
                continue
            seen.add(cu)

            # early allow/deny filter on URL
            if not allowed(cu, allow, deny):
                continue

            downloaded, path = download_pdf(cu)
            if downloaded and path:
                print("saved:", path)
                time.sleep(SLEEP_BETWEEN_DOWNLOADS)
            elif path:
                print("skip (exists):", path)

def main():
    if not os.path.exists(SEEDS_FILE):
        print(f"Missing {SEEDS_FILE}. Create it with one JSON object per line.")
        return

    with open(SEEDS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                seed = json.loads(line)
            except json.JSONDecodeError as e:
                print("seeds.jsonl parse error:", e)
                continue
            process_seed(seed)

if __name__ == "__main__":
    main()