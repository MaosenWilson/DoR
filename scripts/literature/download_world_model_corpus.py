#!/usr/bin/env python3
"""Download and verify open-access PDFs from a finalized paper manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import subprocess
import time
import urllib.request
from pathlib import Path


def filename(record: dict) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", record["title"]).strip("_")[:150]
    return f"{record['id']}_{stem}.pdf"


def candidates(record: dict) -> list[str]:
    urls = []
    for url in record.get("pdf_candidates", []):
        if not url:
            continue
        url = url.replace("http://", "https://")
        if "arxiv.org/abs/" in url:
            url = url.replace("/abs/", "/pdf/")
        urls.append(url)
        if "openreview.net/forum?id=" in url:
            urls.append(url.replace("/forum?id=", "/pdf?id="))
        if "papers.nips.cc" in url and "-Abstract-Conference.html" in url:
            urls.append(url.replace("-Abstract-Conference.html", "-Paper-Conference.pdf"))
        if "proceedings.mlr.press/" in url and url.endswith(".html"):
            stem = url.rsplit("/", 1)[-1][:-5]
            urls.append(url.rsplit("/", 1)[0] + f"/{stem}/{stem}.pdf")
        if "openaccess.thecvf.com/content/" in url and "/html/" in url:
            urls.append(url.replace("/html/", "/papers/").replace("_paper.html", "_paper.pdf"))
    if record.get("arxiv_id"):
        urls.append(f"https://arxiv.org/pdf/{record['arxiv_id']}")
    return list(dict.fromkeys(urls))


def verify(path: Path) -> tuple[bool, int]:
    if not path.exists() or path.stat().st_size < 20_000:
        return False, 0
    if path.read_bytes()[:5] != b"%PDF-":
        return False, 0
    try:
        result = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True, timeout=20)
        match = re.search(r"^Pages:\s+(\d+)", result.stdout, re.M)
        pages = int(match.group(1)) if match else 0
        return result.returncode == 0 and pages > 0, pages
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return True, 0


def download_one(record: dict, root: Path, timeout: int) -> dict:
    directory = root / record["venue"] / str(record["year"])
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / filename(record)
    valid, pages = verify(target)
    if valid:
        return {"id": record["id"], "status": "downloaded", "path": str(target), "bytes": target.stat().st_size, "pages": pages, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "source_url": "resume"}
    errors = []
    for url in candidates(record):
        temp = target.with_suffix(".part")
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 DoR-AAAI2027-literature-audit"})
            with urllib.request.urlopen(request, timeout=timeout) as response, temp.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
            os.replace(temp, target)
            valid, pages = verify(target)
            if not valid:
                target.unlink(missing_ok=True)
                errors.append(f"not_pdf:{url}")
                continue
            return {"id": record["id"], "status": "open_access_downloaded", "path": str(target), "bytes": target.stat().st_size, "pages": pages, "sha256": hashlib.sha256(target.read_bytes()).hexdigest(), "source_url": url}
        except Exception as error:
            temp.unlink(missing_ok=True)
            errors.append(f"{type(error).__name__}:{url}")
    return {"id": record["id"], "status": "no_open_access_pdf_found", "path": "", "bytes": 0, "pages": 0, "sha256": "", "source_url": "", "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--pdf_dir", required=True)
    parser.add_argument("--status", required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    records = json.loads(Path(args.manifest).read_text())["papers"]
    if args.limit:
        records = records[: args.limit]
    root = Path(args.pdf_dir)
    status_path = Path(args.status)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    prior = {}
    if status_path.exists():
        prior = {row["id"]: row for row in json.loads(status_path.read_text()).get("papers", [])}
    started = time.time()
    results = dict(prior)
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(download_one, record, root, args.timeout): record for record in records}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = futures[future]
            try:
                result = future.result()
            except Exception as error:
                result = {"id": record["id"], "status": "download_error", "errors": [repr(error)]}
            results[result["id"]] = result
            elapsed = time.time() - started
            eta = elapsed / index * (len(records) - index) if index else 0
            mb = result.get("bytes", 0) / 1024 / 1024
            print(f"[download] {index:3d}/{len(records)} elapsed={elapsed/60:6.1f}m eta={eta/60:6.1f}m {result['status']:28s} {mb:7.1f}MB {record['title'][:60]}", flush=True)
            ordered = [results[r["id"]] for r in records if r["id"] in results]
            status_path.write_text(json.dumps({"count": len(ordered), "papers": ordered}, ensure_ascii=False, indent=2) + "\n")
    ordered = [results[r["id"]] for r in records]
    downloaded = sum(row["status"] in {"downloaded", "open_access_downloaded"} for row in ordered)
    print(f"[done] downloaded={downloaded}/{len(records)} bytes={sum(r.get('bytes',0) for r in ordered)/1024**3:.2f}GiB status={status_path}")


if __name__ == "__main__":
    main()
