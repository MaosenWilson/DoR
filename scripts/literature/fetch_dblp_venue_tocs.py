#!/usr/bin/env python3
"""Download DBLP main-proceedings tables of contents for target venue-years."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


EDITIONS = {
    "AAAI": ("aaai", "aaai", (2023, 2024, 2025, 2026)),
    "CVPR": ("cvpr", "cvpr", (2023, 2024, 2025, 2026)),
    "ICCV": ("iccv", "iccv", (2023, 2025)),
    "ECCV": ("eccv", "eccv", (2024,)),
    "NeurIPS": ("nips", "neurips", (2023, 2024, 2025)),
    "ICLR": ("iclr", "iclr", (2023, 2024, 2025, 2026)),
    "ICML": ("icml", "icml", (2023, 2024, 2025, 2026)),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    statuses = []
    total = sum(len(years) for _, _, years in EDITIONS.values())
    done = 0
    for venue, (folder, stem, years) in EDITIONS.items():
        for year in years:
            done += 1
            destination = output / f"{venue}_{year}.xml"
            url = f"https://dblp.org/db/conf/{folder}/{stem}{year}.xml"
            command = [
                "curl",
                "-L",
                "--fail",
                "--retry",
                "3",
                "--connect-timeout",
                "20",
                "--silent",
                "--show-error",
                url,
                "-o",
                str(destination),
            ]
            result = subprocess.run(command, check=False)
            valid = destination.exists() and destination.stat().st_size > 500
            if result.returncode != 0 or not valid:
                destination.unlink(missing_ok=True)
                status = "not_available"
            else:
                status = "downloaded"
            statuses.append((venue, year, url, status))
            print(f"[TOC {done:02d}/{total}] {venue} {year}: {status}", flush=True)
    with (output / "manifest.tsv").open("w") as handle:
        handle.write("venue\tyear\turl\tstatus\n")
        for row in statuses:
            handle.write("\t".join(map(str, row)) + "\n")


if __name__ == "__main__":
    main()
