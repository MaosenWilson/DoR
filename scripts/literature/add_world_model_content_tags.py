#!/usr/bin/env python3
"""Add standardized title-derived content tags to the numbered index and DOCX."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from package_numbered_world_models import infer_content_tags, make_docx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index_json", required=True)
    parser.add_argument("--index_tsv", required=True)
    parser.add_argument("--docx", required=True)
    args = parser.parse_args()

    index_path = Path(args.index_json)
    payload = json.loads(index_path.read_text())
    by_number = {}
    for row in payload["papers"]:
        row["content_tags"] = infer_content_tags(row["title"])
        by_number[row["display_no"]] = row["content_tags"]
    index_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    tsv_path = Path(args.index_tsv)
    with tsv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
        fieldnames = list(rows[0].keys())
    if "内容分类" not in fieldnames:
        fieldnames.append("内容分类")
    for row in rows:
        row["内容分类"] = by_number[row["编号"]]
    with tsv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    make_docx(payload["papers"], Path(args.docx), payload["downloaded"])
    counts = {}
    for row in payload["papers"]:
        for tag in row["content_tags"].split(" · "):
            counts[tag] = counts.get(tag, 0) + 1
    print(f"[done] tagged={len(payload['papers'])} categories={len(counts)}")
    for tag, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {tag}: {count}")


if __name__ == "__main__":
    main()
