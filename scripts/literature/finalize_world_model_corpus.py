#!/usr/bin/env python3
"""Merge official supplements, audit semantics, and write the final corpus manifest."""

from __future__ import annotations

import argparse
import csv
import difflib
import html
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path


WORLD_RE = re.compile(r"\bworld[ -]models?\b|\bworld[ -]modell?ing\b", re.I)
METHOD_RE = re.compile(
    r"\b(we (?:propose|present|introduce|develop)|our (?:method|framework|model)|"
    r"world model (?:that|which|is|learns|predicts|simulates)|based on (?:a |the )?world model)\b",
    re.I,
)
MANUAL_EXCLUSIONS = {
    "Plug-and-Play Cooperative Navigation: From Single-Agent Navigation Fields to Graph- Maintaining Distributed MAS Controllers": "sphere-world analytic geometry, not a learned world model",
    "Achieving Secure On-Orbit Anomaly Identification and Query of Wind Turbines": "cryptographic real/ideal-world security model",
    "Real-Time Monitoring and Safety Scheduling of Intelligent Elevators Based on Biometric Data Analysis": "YOLO-World detector name",
    "For Overall Nighttime Visibility: Integrate Irregular Glow Removal With Glow-Aware Enhancement": "physical-world optics model, not a learned world model",
    "LMAgent: A Large-scale Multimodal Agents Society for Multi-user Simulation": "small-world graph model",
}


def clean(value: object) -> str:
    return re.sub(r"\s+", " ", html.unescape(str(value or ""))).strip()


def norm_title(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def request_json(url: str, retries: int = 4) -> dict:
    headers = {"User-Agent": "DoR-AAAI2027-literature-audit/1.0"}
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=60) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == retries:
                return {}
            time.sleep(2 ** attempt)
    return {}


def s2_lookup(title: str, cache_dir: Path) -> dict:
    cache = cache_dir / (norm_title(title)[:100] + ".json")
    if cache.exists():
        return json.loads(cache.read_text())
    fields = "title,abstract,year,venue,externalIds,openAccessPdf,url,authors,publicationDate"
    params = urllib.parse.urlencode({"query": title, "limit": 5, "fields": fields})
    payload = request_json("https://api.semanticscholar.org/graph/v1/paper/search?" + params)
    target = norm_title(title)
    best: dict = {}
    best_score = 0.0
    for item in payload.get("data", []):
        score = difflib.SequenceMatcher(None, target, norm_title(clean(item.get("title")))).ratio()
        if score > best_score:
            best, best_score = item, score
    if best_score < 0.94:
        best = {}
    cache.write_text(json.dumps(best, ensure_ascii=False, indent=2) + "\n")
    time.sleep(1.05)
    return best


def cvf_pdf(page: str) -> str:
    return page.replace("/html/", "/papers/").replace("_paper.html", "_paper.pdf")


def supplement_record(row: dict) -> dict:
    page = row.get("official_page", "")
    openreview_id = row.get("openreview_id", "")
    arxiv_id = row.get("arxiv_id", "")
    pdfs = []
    if "/content/CVPR" in page:
        pdfs.append(cvf_pdf(page))
    if openreview_id:
        page = f"https://openreview.net/forum?id={openreview_id}"
        pdfs.append(f"https://openreview.net/pdf?id={openreview_id}")
    if arxiv_id:
        pdfs.append(f"https://arxiv.org/pdf/{arxiv_id}")
    return {
        "title": row["title"],
        "abstract": "",
        "year": int(row["year"]),
        "date": "",
        "venue": row["venue"],
        "venue_raw": f"{row['venue']} 2026 official main conference",
        "authors": [],
        "doi": "",
        "arxiv_id": arxiv_id,
        "dblp_key": "",
        "paper_url": page,
        "pdf_candidates": pdfs,
        "match_location": "title" if WORLD_RE.search(row["title"]) else "abstract",
        "sources": ["official-2026-supplement"],
        "source_ids": [openreview_id or page],
        "venue_evidence": "official-main-conference-page",
    }


def enrich(record: dict, item: dict) -> None:
    if not item:
        return
    if not record.get("abstract"):
        record["abstract"] = clean(item.get("abstract"))
    if not record.get("authors"):
        record["authors"] = [clean(a.get("name")) for a in item.get("authors", [])]
    if not record.get("date"):
        record["date"] = clean(item.get("publicationDate"))
    external = item.get("externalIds") or {}
    if not record.get("doi"):
        record["doi"] = clean(external.get("DOI"))
    if not record.get("arxiv_id"):
        record["arxiv_id"] = clean(external.get("ArXiv"))
    candidates = list(record.get("pdf_candidates", []))
    oa = item.get("openAccessPdf") or {}
    if oa.get("url"):
        candidates.append(oa["url"])
    if record.get("arxiv_id"):
        candidates.append(f"https://arxiv.org/pdf/{record['arxiv_id']}")
    record["pdf_candidates"] = list(dict.fromkeys(url for url in candidates if url))
    record["sources"] = list(dict.fromkeys(record.get("sources", []) + ["Semantic Scholar enrichment"]))


def classify(record: dict) -> tuple[str, str, str]:
    title = record.get("title", "")
    abstract = record.get("abstract", "")
    location = "title+abstract" if WORLD_RE.search(title) and WORLD_RE.search(abstract) else (
        "title" if WORLD_RE.search(title) else "abstract"
    )
    sentences = re.split(r"(?<=[.!?])\s+", abstract or title)
    matched = next((sentence for sentence in sentences if WORLD_RE.search(sentence)), title)
    if WORLD_RE.search(title) or METHOD_RE.search(abstract):
        category = "method_or_object_core"
    else:
        category = "context_or_related_mention"
    return location, clean(matched)[:900], category


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--validated", required=True)
    parser.add_argument("--supplement", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--no_enrich", action="store_true")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = out_dir / "s2_enrichment_cache"
    cache_dir.mkdir(exist_ok=True)

    source = json.loads(Path(args.validated).read_text())
    supplement = json.loads(Path(args.supplement).read_text())
    excluded = []
    merged: dict[str, dict] = {}
    for row in source["papers"]:
        if row["title"] in MANUAL_EXCLUSIONS:
            excluded.append({**row, "exclusion_reason": MANUAL_EXCLUSIONS[row["title"]]})
            continue
        merged[norm_title(row["title"])] = dict(row)
    for row in supplement["papers"]:
        key = norm_title(row["title"])
        if key not in merged:
            merged[key] = supplement_record(row)
        else:
            extra = supplement_record(row)
            target = merged[key]
            target["pdf_candidates"] = list(dict.fromkeys(target.get("pdf_candidates", []) + extra["pdf_candidates"]))
            target["sources"] = list(dict.fromkeys(target.get("sources", []) + extra["sources"]))
            target["venue_evidence"] = extra["venue_evidence"]

    records = list(merged.values())
    if not args.no_enrich:
        pending = [r for r in records if not r.get("abstract") or not r.get("pdf_candidates")]
        for index, record in enumerate(pending, 1):
            print(f"[enrich] {index:3d}/{len(pending)} {record['title'][:82]}", flush=True)
            enrich(record, s2_lookup(record["title"], cache_dir))

    final = []
    for record in records:
        if not WORLD_RE.search(record.get("title", "") + " " + record.get("abstract", "")):
            excluded.append({**record, "exclusion_reason": "world-model phrase absent after enrichment"})
            continue
        location, sentence, category = classify(record)
        record["match_location"] = location
        record["matched_sentence"] = sentence
        record["abstract_category"] = category
        final.append(record)
    final.sort(key=lambda r: (int(r["year"]), r["venue"], r["title"].lower()))
    for index, record in enumerate(final, 1):
        record["id"] = f"WM{index:03d}"

    payload = {
        "scope": {
            "years": [2023, 2026],
            "cutoff": supplement["cutoff"],
            "inclusion": "Main venue; title or abstract explicitly contains world model(s)/world modeling.",
            "exclusion": "Workshops, withdrawn/submitted-only papers, and semantic homonyms.",
        },
        "count": len(final),
        "excluded_count": len(excluded),
        "papers": final,
    }
    (out_dir / "final_manifest.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    (out_dir / "semantic_exclusions.json").write_text(json.dumps(excluded, ensure_ascii=False, indent=2) + "\n")
    fields = ["id", "year", "venue", "title", "abstract_category", "match_location", "doi", "arxiv_id", "paper_url", "abstract"]
    with (out_dir / "final_manifest.tsv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(final)
    print(f"[done] final={len(final)} exclusions={len(excluded)} missing_abstract={sum(not r.get('abstract') for r in final)}")


if __name__ == "__main__":
    main()
