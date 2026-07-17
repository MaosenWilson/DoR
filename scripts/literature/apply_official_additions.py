#!/usr/bin/env python3
"""Apply manually audited official additions without renumbering existing records."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def norm(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--additions", required=True)
    args = parser.parse_args()
    path = Path(args.manifest)
    payload = json.loads(path.read_text())
    additions = json.loads(Path(args.additions).read_text())["papers"]
    by_title = {norm(row["title"]): row for row in payload["papers"]}
    next_id = max(int(row["id"][2:]) for row in payload["papers"]) + 1
    added = overridden = 0
    for item in additions:
        if item["action"] == "override":
            row = by_title[norm(item["match_title"])]
            row["pdf_candidates"] = list(dict.fromkeys(row.get("pdf_candidates", []) + [item["official_pdf"]]))
            row["sources"] = list(dict.fromkeys(row.get("sources", []) + ["official manual audit"]))
            overridden += 1
            continue
        key = norm(item["title"])
        if key in by_title:
            continue
        row = {
            "id": f"WM{next_id:03d}",
            "title": item["title"],
            "abstract": "",
            "year": item["year"],
            "date": "",
            "venue": item["venue"],
            "venue_raw": f"{item['venue']} official main conference",
            "authors": [],
            "doi": "",
            "arxiv_id": "",
            "dblp_key": "",
            "paper_url": item["official_page"],
            "pdf_candidates": [item["official_pdf"]],
            "match_location": item["match_location"],
            "sources": ["official manual audit"],
            "source_ids": [item["official_page"]],
            "venue_evidence": "official-main-conference-page",
            "matched_sentence": item["title"],
            "abstract_category": "method_or_object_core" if item["match_location"] != "abstract" else "context_or_related_mention",
        }
        payload["papers"].append(row)
        by_title[key] = row
        next_id += 1
        added += 1
    payload["count"] = len(payload["papers"])
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(f"[done] added={added} overridden={overridden} total={payload['count']}")


if __name__ == "__main__":
    main()
