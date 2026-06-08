"""
compare_segmenters.py
=====================
Side-by-side human-review comparison of NLTK / spaCy / SaT segmenters.

Run::

    python compare_segmenters.py

Output: compare_segmenters.html — open in any browser.

No metrics, no scoring, no gold comparison.  For visual review only.
"""
from __future__ import annotations

# ── Configuration (edit freely) ───────────────────────────────────────────
N_DOCS      = 25                               # docs to display
SEED        = 42                               # reproducibility
DATA_FILE   = "data/raw/sample/sample.jsonl"  # raw job-ad source
OUTPUT_HTML = "compare_segmenters.html"
SAT_MODEL   = "sat-3l-sm"
# ──────────────────────────────────────────────────────────────────────────

import html as _html
import json
import os
import random
import sys
from statistics import mode as _mode

# Fix stale SSL_CERT_FILE that some conda envs point to a missing path.
_cert = os.environ.get("SSL_CERT_FILE", "")
if _cert and not os.path.exists(_cert):
    try:
        import certifi
        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass

from preprocessing.cleaner import clean_description
from segmenters.nltk_splitter import nltk_split
from segmenters import SpacySegmenter, SatSegmenter


# ── 1. Load & sample ──────────────────────────────────────────────────────

print(f"Loading {DATA_FILE} …")
with open(DATA_FILE, encoding="utf-8") as fh:
    records = [json.loads(line) for line in fh if line.strip()]

random.seed(SEED)
random.shuffle(records)

print(f"Cleaning (keeping first {N_DOCS} non-empty results) …")
docs: list[dict] = []
skipped = 0
for rec in records:
    if len(docs) >= N_DOCS:
        break
    raw = rec.get("text") or rec.get("description") or ""
    cleaned = clean_description(raw)
    if not cleaned.strip():
        skipped += 1
        continue
    docs.append({
        "id":      rec.get("usajobsControlNumber") or rec.get("job_id") or "?",
        "title":   rec.get("title") or "",
        "cleaned": cleaned,
    })

print(f"  {len(docs)} docs kept, {skipped} skipped (empty after cleaning).")

if not docs:
    sys.exit("No usable docs after cleaning — check DATA_FILE path.")


# ── 2. Instantiate segmenters ──────────────────────────────────────────────

print("Loading spaCy (en_core_web_sm) …")
spacy_seg = SpacySegmenter()

print(f"Loading SaT '{SAT_MODEL}' (downloads ~200 MB on first use) …")
sat_seg = SatSegmenter(model=SAT_MODEL)


# ── 3. Segment ────────────────────────────────────────────────────────────

print("Segmenting …")
for i, doc in enumerate(docs):
    txt = doc["cleaned"]
    doc["nltk"]  = nltk_split(txt)
    doc["spacy"] = spacy_seg.segment(txt)
    doc["sat"]   = sat_seg.segment(txt)
    print(
        f"  {i+1:>2}/{len(docs)}"
        f"  NLTK={len(doc['nltk']):<3}"
        f"  spaCy={len(doc['spacy']):<3}"
        f"  SaT={len(doc['sat']):<3}"
        f"  — {doc['title'][:60]}"
    )


# ── 4. HTML rendering ─────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; }
body {
    font-family: system-ui, -apple-system, sans-serif;
    margin: 2rem auto; max-width: 1500px; padding: 0 1.2rem;
    background: #f4f6f8; color: #1a1a1a; font-size: 14px;
}
h1 { font-size: 1.4rem; border-bottom: 3px solid #2a4080; padding-bottom: .5rem; }
.meta { color: #555; font-size: .85rem; margin-bottom: 2rem; line-height: 1.8; }
.doc-block {
    background: #fff; border: 1px solid #d0d7de; border-radius: 8px;
    margin-bottom: 2rem; padding: 1rem 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,.06);
}
.doc-block h2 {
    margin: 0 0 .6rem 0; font-size: 1rem; color: #0d2060;
    display: flex; align-items: baseline; gap: .5rem; flex-wrap: wrap;
}
.doc-id { font-size: .78rem; font-weight: normal; color: #666; }
details { margin-bottom: .8rem; }
summary { cursor: pointer; color: #2a4080; font-size: .82rem; user-select: none; }
pre.cleaned {
    background: #f6f8fa; border: 1px solid #e0e4ea; border-radius: 4px;
    padding: .6rem .8rem; white-space: pre-wrap; word-break: break-word;
    font-size: .78rem; max-height: 180px; overflow-y: auto; margin: .4rem 0 0;
    color: #333;
}
table { border-collapse: collapse; width: 100%; table-layout: fixed; }
thead tr { background: #2a4080; }
th {
    padding: .45rem .6rem; text-align: left; font-size: .82rem;
    color: #fff; width: 33.33%;
}
th.diff { background: #8b1a1a; }
td {
    vertical-align: top; padding: .28rem .5rem; font-size: .8rem;
    border-bottom: 1px solid #eaecef; width: 33.33%;
    word-break: break-word;
}
tbody tr:last-child td { border-bottom: none; }
tbody tr:hover td { background: #eef2ff; }
.idx { color: #aaa; font-size: .72rem; margin-right: .25rem; }
.empty-col { color: #bbb; font-style: italic; }
"""


def _e(s: object) -> str:
    return _html.escape(str(s))


def _render_doc(n: int, total: int, doc: dict) -> str:
    nltk_segs  = doc["nltk"]
    spacy_segs = doc["spacy"]
    sat_segs   = doc["sat"]

    counts = [len(nltk_segs), len(spacy_segs), len(sat_segs)]
    all_agree = len(set(counts)) == 1
    try:
        consensus = _mode(counts)
    except Exception:
        consensus = counts[0]

    def th_html(label: str, count: int) -> str:
        cls = "" if all_agree or count == consensus else ' class="diff"'
        return f"<th{cls}>{_e(label)} ({count})</th>"

    max_rows = max(counts) if counts else 0
    rows: list[str] = []
    for i in range(max_rows):
        def cell(segs: list[str], idx: int) -> str:
            if idx < len(segs):
                return (f"<td><span class='idx'>{idx+1}.</span>"
                        f"{_e(segs[idx])}</td>")
            return "<td></td>"
        rows.append(
            f"<tr>{cell(nltk_segs,i)}{cell(spacy_segs,i)}{cell(sat_segs,i)}</tr>"
        )

    if not rows:
        rows = ["<tr><td class='empty-col'>—</td>"
                "<td class='empty-col'>—</td>"
                "<td class='empty-col'>—</td></tr>"]

    title_str = _e(doc["title"]) if doc["title"] else "<em>no title</em>"
    cleaned_e = _e(doc["cleaned"])

    return (
        f'<div class="doc-block">'
        f'<h2>{n}/{total} &mdash; {title_str}'
        f' <span class="doc-id">id: {_e(doc["id"])}'
        f' &bull; {len(doc["cleaned"])} chars</span></h2>'
        f"<details><summary>Cleaned text</summary>"
        f"<pre class='cleaned'>{cleaned_e}</pre></details>"
        f"<table><thead><tr>"
        f"{th_html('NLTK', len(nltk_segs))}"
        f"{th_html('spaCy', len(spacy_segs))}"
        f"{th_html('SaT', len(sat_segs))}"
        f"</tr></thead><tbody>{''.join(rows)}</tbody></table>"
        f"</div>"
    )


def _render_html(docs: list[dict]) -> str:
    cards = "\n".join(_render_doc(i + 1, len(docs), d) for i, d in enumerate(docs))
    return (
        '<!DOCTYPE html>\n<html lang="en">\n<head>\n'
        '<meta charset="utf-8">\n'
        '<meta name="viewport" content="width=device-width,initial-scale=1">\n'
        f"<title>Segmenter Comparison</title>\n"
        f"<style>{_CSS}</style>\n</head>\n<body>\n"
        f"<h1>Segmenter Comparison &mdash; USAJOBS</h1>\n"
        f'<p class="meta">'
        f"{len(docs)} docs &bull; seed={SEED} &bull; "
        f"SaT model: <code>{_e(SAT_MODEL)}</code><br>"
        f"Source: <code>{_e(DATA_FILE)}</code><br>"
        f"<em>Header counts in dark red = that segmenter disagrees with the majority count.</em>"
        f"</p>\n"
        f"{cards}\n</body>\n</html>"
    )


# ── 5. Write output ───────────────────────────────────────────────────────

print(f"\nWriting {OUTPUT_HTML} …")
html_out = _render_html(docs)
with open(OUTPUT_HTML, "w", encoding="utf-8") as fh:
    fh.write(html_out)

print(f"Done.  Open  {OUTPUT_HTML}  in a browser to review.")
