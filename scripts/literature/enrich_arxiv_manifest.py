#!/usr/bin/env python3
"""Enrich missing abstracts/PDF links via exact-title arXiv API matches."""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path


ATOM = {"a": "http://www.w3.org/2005/Atom"}


def clean(value: str) -> str:
    return re.sub(r"\s+", " ", html.unescape(value or "")).strip()


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def query(title: str, cache: Path, arxiv_id: str = "") -> dict:
    suffix = f"_id_{arxiv_id}" if arxiv_id else ""
    path = cache / (norm(title)[:110] + suffix + ".json")
    if path.exists():
        cached = json.loads(path.read_text())
        if cached and not cached.get("error"):
            return cached
    query_args = {"start": 0, "max_results": 5}
    if arxiv_id:
        query_args["id_list"] = arxiv_id
    else:
        query_args["search_query"] = f'ti:"{title}"'
    params = urllib.parse.urlencode(query_args)
    url = "https://export.arxiv.org/api/query?" + params
    result = {}
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "DoR-AAAI2027-literature-audit/1.0"})
        with urllib.request.urlopen(request, timeout=90) as response:
            root = ET.fromstring(response.read())
        best_score = 0.0
        for entry in root.findall("a:entry", ATOM):
            candidate = clean(entry.findtext("a:title", "", ATOM))
            score = difflib.SequenceMatcher(None, norm(title), norm(candidate)).ratio()
            if score <= best_score:
                continue
            arxiv_url = clean(entry.findtext("a:id", "", ATOM))
            arxiv_id = arxiv_url.rsplit("/", 1)[-1].split("v", 1)[0]
            result = {
                "title": candidate,
                "abstract": clean(entry.findtext("a:summary", "", ATOM)),
                "arxiv_id": arxiv_id,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}",
                "score": score,
            }
            best_score = score
        if best_score < 0.94:
            result = {}
    except Exception as error:
        result = {"error": repr(error)}
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    time.sleep(3.1)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--download_status")
    parser.add_argument("--cache", required=True)
    args = parser.parse_args()
    path = Path(args.manifest)
    payload = json.loads(path.read_text())
    failed = set()
    if args.download_status and Path(args.download_status).exists():
        status = json.loads(Path(args.download_status).read_text())
        failed = {row["id"] for row in status["papers"] if "downloaded" not in row["status"]}
    pending = [row for row in payload["papers"] if not row.get("abstract") or row["id"] in failed]
    cache = Path(args.cache)
    cache.mkdir(parents=True, exist_ok=True)
    matched = 0
    for index, row in enumerate(pending, 1):
        print(f"[arXiv] {index:3d}/{len(pending)} {row['title'][:86]}", flush=True)
        result = query(row["title"], cache, row.get("arxiv_id", "") if not row.get("abstract") else "")
        if not result or result.get("error"):
            continue
        matched += 1
        if not row.get("abstract"):
            row["abstract"] = result["abstract"]
        if not row.get("arxiv_id"):
            row["arxiv_id"] = result["arxiv_id"]
        row["pdf_candidates"] = list(dict.fromkeys(row.get("pdf_candidates", []) + [result["pdf_url"]]))
        row["sources"] = list(dict.fromkeys(row.get("sources", []) + ["arXiv exact-title enrichment"]))
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"[done] pending={len(pending)} matched={matched} missing_abstract={sum(not r.get('abstract') for r in payload['papers'])}")


if __name__ == "__main__":
    main()
