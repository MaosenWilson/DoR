#!/usr/bin/env python3
"""Rebuild a continuous numbered package from downloaded rows in the TSV index."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

from package_numbered_world_models import make_docx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_tsv", required=True)
    parser.add_argument("--source_pdf_dir", required=True)
    parser.add_argument("--output_pdf_dir", required=True)
    parser.add_argument("--output_tsv", required=True)
    parser.add_argument("--output_json", required=True)
    parser.add_argument("--output_docx", required=True)
    args = parser.parse_args()

    source_dir = Path(args.source_pdf_dir)
    output_dir = Path(args.output_pdf_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    with Path(args.source_tsv).open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle, delimiter="\t"))

    rows = []
    for source_row in source_rows:
        filename = source_row["PDF文件"].strip()
        if source_row["全文状态"] != "已下载" or not filename:
            continue
        source = source_dir / filename
        if not source.is_file():
            raise FileNotFoundError(source)
        number = f"{len(rows) + 1:03d}"
        suffix = filename.split("_", 1)[1]
        target = output_dir / f"{number}_{suffix}"
        os.link(source, target)
        rows.append(
            {
                "display_no": number,
                "year": int(source_row["年份"]),
                "venue": source_row["来源"],
                "title": source_row["文献题名"],
                "flat_pdf_path": str(target),
                "download_status": "downloaded",
            }
        )

    output_tsv = Path(args.output_tsv)
    with output_tsv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(["编号", "年份", "来源", "文献题名", "全文状态", "PDF文件"])
        for row in rows:
            writer.writerow(
                [
                    row["display_no"],
                    row["year"],
                    row["venue"],
                    row["title"],
                    "已下载",
                    Path(row["flat_pdf_path"]).name,
                ]
            )

    payload = {"count": len(rows), "downloaded": len(rows), "missing": 0, "papers": rows}
    Path(args.output_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    make_docx(rows, Path(args.output_docx), len(rows))
    print(f"[done] continuously numbered {len(rows)} downloaded papers")


if __name__ == "__main__":
    main()
