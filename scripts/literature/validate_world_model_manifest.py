#!/usr/bin/env python3
"""Validate venue/year and remove lexical false positives from a paper manifest."""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import re
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

from search_world_models_2023plus import (
    DBLP_KEY_RULES,
    YEAR_MAX,
    YEAR_MIN,
    _clean,
    _norm_title,
)


CONFERENCES = {"AAAI", "CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR", "ICML"}
JOURNALS = {"TPAMI", "TIP", "TNNLS", "TMM", "TCSVT", "Other IEEE Transactions"}
SEMANTIC_RE = re.compile(
    r"\b(agent|environment|dynamics?|transition|reinforcement|control|planning|imagin|predict|"
    r"generative|simulat|video|robot|autonomous|embodied|state|physical|action|trajectory|"
    r"representation|reasoning)\w*\b",
    re.I,
)
WORLD_RE = re.compile(r"\bworld[ -]models?\b|\bworld[ -]modell?ing\b", re.I)
LEXICAL_PREFIXES = (
    "real-",
    "real ",
    "open-",
    "open ",
    "closed-",
    "closed ",
    "multi-",
    "multi ",
)


def _request_json(url: str, retries: int = 5) -> dict:
    headers = {"User-Agent": "DoR-AAAI2027-literature-audit/1.0 (mailto:research@example.com)"}
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=45) as response:
                return json.load(response)
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _venue_from_key(key: str) -> str:
    for venue, pattern in DBLP_KEY_RULES:
        if pattern.search(key):
            return venue
    return ""


def _valid_world_matches(text: str) -> list[re.Match]:
    matches = []
    lower = text.lower()
    for match in WORLD_RE.finditer(text):
        prefix = lower[max(0, match.start() - 12) : match.start()]
        if any(prefix.endswith(value) for value in LEXICAL_PREFIXES):
            continue
        matches.append(match)
    return matches


def _semantic_evidence(record: dict) -> tuple[bool, str, str]:
    title = record.get("title", "")
    abstract = record.get("abstract", "")
    title_matches = _valid_world_matches(title)
    abstract_matches = _valid_world_matches(abstract)
    if not title_matches and not abstract_matches:
        return False, "lexical_collision", ""
    if not title_matches and not SEMANTIC_RE.search(abstract):
        return False, "no_world_model_semantics", ""
    text = abstract or title
    for sentence in re.split(r"(?<=[.!?])\s+", text):
        if _valid_world_matches(sentence):
            return True, "", sentence[:700]
    return True, "", title[:700]


def _calendar_valid(venue: str, year: int) -> bool:
    if not (YEAR_MIN <= year <= YEAR_MAX):
        return False
    if venue == "ECCV" and year % 2:
        return False
    if venue == "ICCV" and not year % 2:
        return False
    if venue == "NeurIPS" and year > 2025:
        return False
    return True


def _official_openalex_evidence(record: dict) -> bool:
    if "OpenAlex" not in record.get("sources", []):
        return False
    raw = record.get("venue_raw", "")
    if re.search(r"workshop|symposium|findings|AIIDE", raw, re.I):
        return False
    venue = record["venue"]
    exact = {
        "AAAI": r"^(?:Proceedings of the )?AAAI Conference on Artificial Intelligence$",
        "CVPR": r"^Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition$",
        "ICCV": r"^(?:Proceedings of the )?IEEE/CVF International Conference on Computer Vision$",
        "ECCV": r"^European Conference on Computer Vision$",
        "NeurIPS": r"^(?:Advances in )?Neural Information Processing Systems$",
        "ICLR": r"^International Conference on Learning Representations$",
        "ICML": r"^(?:Proceedings of the )?International Conference on Machine Learning$",
        "TPAMI": r"^IEEE Transactions on Pattern Analysis and Machine Intelligence$",
        "TIP": r"^IEEE Transactions on Image Processing$",
        "TNNLS": r"^IEEE Transactions on Neural Networks and Learning Systems$",
        "TMM": r"^IEEE Transactions on Multimedia$",
        "TCSVT": r"^IEEE Transactions on Circuits and Systems for Video Technology$",
        "Other IEEE Transactions": r"^IEEE Transactions on ",
    }[venue]
    return bool(re.search(exact, raw, re.I))


def _dblp_search(title: str, cache_dir: Path) -> list[dict]:
    digest = hashlib.sha1(title.encode()).hexdigest()
    cache = cache_dir / f"{digest}.json"
    if cache.exists():
        payload = json.loads(cache.read_text())
    else:
        params = {"q": title, "format": "json", "h": "10"}
        url = "https://dblp.org/search/publ/api?" + urllib.parse.urlencode(params)
        try:
            payload = _request_json(url)
        except Exception as error:
            print(f"[DBLP warning] {type(error).__name__}: {title[:80]}", flush=True)
            return []
        cache.write_text(json.dumps(payload))
        time.sleep(0.18)
    return ((payload.get("result") or {}).get("hits") or {}).get("hit") or []


def _best_dblp_match(record: dict, cache_dir: Path) -> dict | None:
    title_norm = _norm_title(record["title"])
    best = None
    best_score = 0.0
    for hit in _dblp_search(record["title"], cache_dir):
        info = hit.get("info") or {}
        candidate_title = _clean(info.get("title"))
        score = difflib.SequenceMatcher(None, title_norm, _norm_title(candidate_title)).ratio()
        key = _clean(info.get("key"))
        venue = _venue_from_key(key)
        if venue != record["venue"] or score < 0.94:
            continue
        if score > best_score:
            best = info
            best_score = score
    return best


def _embedded_dblp_match(record: dict) -> dict | None:
    key = _clean(record.get("dblp_key"))
    if _venue_from_key(key) != record["venue"]:
        return None
    year_match = re.search(r"(\d{2})$", key)
    year = 2000 + int(year_match.group(1)) if year_match else int(record["year"])
    return {"key": key, "year": year, "ee": []}


def _merge_validated(records: list[dict]) -> list[dict]:
    result = {}
    for record in records:
        key = _norm_title(record["title"])
        if key not in result:
            result[key] = record
            continue
        target = result[key]
        if len(record.get("abstract", "")) > len(target.get("abstract", "")):
            target["abstract"] = record["abstract"]
            target["matched_sentence"] = record["matched_sentence"]
        for field in ("doi", "arxiv_id", "dblp_key", "paper_url"):
            if not target.get(field) and record.get(field):
                target[field] = record[field]
        target["pdf_candidates"] = list(
            dict.fromkeys(target.get("pdf_candidates", []) + record.get("pdf_candidates", []))
        )
        target["sources"] = list(dict.fromkeys(target["sources"] + record["sources"]))
    papers = list(result.values())
    papers.sort(key=lambda row: (row["year"], row["venue"], row["title"].lower()))
    for index, row in enumerate(papers, 1):
        row["id"] = f"WM{index:03d}"
    return papers


def _load_toc_index(directory: Path | None) -> dict:
    if directory is None:
        return {"exact": {}, "by_venue": {}}
    entries = []
    for path in sorted(directory.glob("*.xml")):
        match = re.match(r"(?P<venue>[^_]+)_(?P<year>\d{4})\.xml$", path.name)
        if not match:
            continue
        try:
            root = ET.parse(path).getroot()
        except ET.ParseError as error:
            print(f"[TOC warning] {path.name}: {error}", flush=True)
            continue
        for item in root.iter("inproceedings"):
            title_node = item.find("title")
            title = _clean("".join(title_node.itertext()) if title_node is not None else "")
            if not title:
                continue
            links = [_clean(node.text) for node in item.findall("ee") if node.text]
            entries.append(
                {
                    "title": title,
                    "title_norm": _norm_title(title),
                    "venue": match.group("venue"),
                    "year": int(match.group("year")),
                    "key": item.get("key") or "",
                    "ee": links,
                }
            )
    exact = {(entry["venue"], entry["title_norm"]): entry for entry in entries}
    by_venue = {}
    for entry in entries:
        by_venue.setdefault(entry["venue"], []).append(entry)
    print(f"[TOC index] entries={len(entries)}", flush=True)
    return {"exact": exact, "by_venue": by_venue}


def _toc_match(record: dict, index: dict) -> dict | None:
    title_norm = _norm_title(record["title"])
    exact = index["exact"].get((record["venue"], title_norm))
    if exact:
        return exact
    best = None
    best_score = 0.0
    for entry in index["by_venue"].get(record["venue"], []):
        score = difflib.SequenceMatcher(None, title_norm, entry["title_norm"]).ratio()
        if score >= 0.965 and score > best_score:
            best = entry
            best_score = score
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--query_dblp", action="store_true")
    parser.add_argument("--dblp_toc_dir")
    args = parser.parse_args()
    source = json.loads(Path(args.manifest).read_text())
    out_dir = Path(args.out_dir)
    cache_dir = out_dir / "dblp_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    toc_entries = _load_toc_index(Path(args.dblp_toc_dir) if args.dblp_toc_dir else None)

    accepted = []
    excluded = []
    papers = source["papers"]
    for index, record in enumerate(papers, 1):
        record = dict(record)
        semantic_ok, reason, sentence = _semantic_evidence(record)
        if not semantic_ok:
            record["exclusion_reason"] = reason
            excluded.append(record)
            continue

        toc_match = _toc_match(record, toc_entries) if record["venue"] in CONFERENCES else None
        match = toc_match or _embedded_dblp_match(record)
        if match is None and args.query_dblp:
            match = _best_dblp_match(record, cache_dir)
        official_openalex = _official_openalex_evidence(record)
        if match:
            record["dblp_key"] = _clean(match.get("key"))
            record["year"] = int(match.get("year") or record["year"])
            record["venue_evidence"] = "DBLP-main-TOC" if toc_match else "DBLP-main"
            ee = match.get("ee") or []
            if isinstance(ee, str):
                ee = [ee]
            record["pdf_candidates"] = list(
                dict.fromkeys(record.get("pdf_candidates", []) + list(ee))
            )
        elif official_openalex:
            record["venue_evidence"] = "OpenAlex-exact-source"
        else:
            record["exclusion_reason"] = "venue_unverified"
            excluded.append(record)
            continue

        if not _calendar_valid(record["venue"], int(record["year"])):
            record["exclusion_reason"] = "impossible_conference_year"
            excluded.append(record)
            continue
        record["matched_sentence"] = sentence
        accepted.append(record)
        if index % 25 == 0:
            print(
                f"[validate] {index}/{len(papers)} accepted={len(accepted)} excluded={len(excluded)}",
                flush=True,
            )

    accepted = _merge_validated(accepted)
    payload = {
        "scope": source["scope"],
        "count": len(accepted),
        "excluded_count": len(excluded),
        "papers": accepted,
    }
    (out_dir / "validated_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    (out_dir / "excluded.json").write_text(
        json.dumps(excluded, ensure_ascii=False, indent=2) + "\n"
    )
    counts = {}
    for row in accepted:
        counts[(row["year"], row["venue"])] = counts.get((row["year"], row["venue"]), 0) + 1
    print(f"[done] accepted={len(accepted)} excluded={len(excluded)}")
    for (year, venue), count in sorted(counts.items()):
        print(f"  {year} {venue:24s} {count:3d}")


if __name__ == "__main__":
    main()
