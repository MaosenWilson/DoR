#!/usr/bin/env python3
"""Build a reproducible 2023+ world-model paper manifest for selected venues."""

from __future__ import annotations

import argparse
import json
import re
import time
import unicodedata
import urllib.parse
import urllib.request
import urllib.error
from pathlib import Path


YEAR_MIN = 2023
YEAR_MAX = 2026
PHRASE_RE = re.compile(
    r"\bworld[ -]models?\b|\bworld[ -]modell?ing\b", re.IGNORECASE
)

VENUE_RULES = (
    (
        "AAAI",
        re.compile(r"^(?:Proceedings of the )?AAAI Conference on Artificial Intelligence$", re.I),
    ),
    ("CVPR", re.compile(r"Computer Vision and Pattern Recognition|\bCVPR\b", re.I)),
    ("ICCV", re.compile(r"International Conference on Computer Vision|\bICCV\b", re.I)),
    ("ECCV", re.compile(r"European Conference on Computer Vision|\bECCV\b", re.I)),
    ("NeurIPS", re.compile(r"Neural Information Processing Systems|\bNeurIPS\b", re.I)),
    ("ICLR", re.compile(r"International Conference on Learning Representations|\bICLR\b", re.I)),
    ("ICML", re.compile(r"International Conference on Machine Learning|\bICML\b", re.I)),
    (
        "TPAMI",
        re.compile(
            r"IEEE Transactions on Pattern Analysis and Machine Intelligence|\bTPAMI\b|\bPAMI\b",
            re.I,
        ),
    ),
    ("TIP", re.compile(r"IEEE Transactions on Image Processing|\bTIP\b", re.I)),
    (
        "TNNLS",
        re.compile(r"IEEE Transactions on Neural Networks and Learning Systems|\bTNNLS\b", re.I),
    ),
    ("TMM", re.compile(r"IEEE Transactions on Multimedia|\bTMM\b", re.I)),
    (
        "TCSVT",
        re.compile(r"IEEE Transactions on Circuits and Systems for Video Technology|\bTCSVT\b", re.I),
    ),
)

DBLP_KEY_RULES = (
    ("AAAI", re.compile(r"^conf/aaai/")),
    ("CVPR", re.compile(r"^conf/cvpr/")),
    ("ICCV", re.compile(r"^conf/iccv/")),
    ("ECCV", re.compile(r"^conf/eccv/")),
    ("NeurIPS", re.compile(r"^conf/nips/")),
    ("ICLR", re.compile(r"^conf/iclr/")),
    ("ICML", re.compile(r"^conf/icml/")),
    ("TPAMI", re.compile(r"^journals/tpami/")),
    ("TIP", re.compile(r"^journals/tip/")),
    ("TNNLS", re.compile(r"^journals/tnn/")),
    ("TMM", re.compile(r"^journals/tmm/")),
    ("TCSVT", re.compile(r"^journals/tcsv/")),
)


def _request_json(url: str, retries: int = 5) -> dict:
    headers = {"User-Agent": "DoR-AAAI2027-literature-audit/1.0 (mailto:research@example.com)"}
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            if error.code == 429 and attempt + 1 < retries:
                delay = int(error.headers.get("Retry-After") or 30 * (attempt + 1))
                print(f"[rate-limit] waiting {delay}s", flush=True)
                time.sleep(delay)
                continue
            raise
        except Exception:
            if attempt + 1 == retries:
                raise
            time.sleep(2**attempt)
    raise RuntimeError("unreachable")


def _abstract(index: dict | None) -> str:
    if not index:
        return ""
    positions = []
    for word, slots in index.items():
        positions.extend((int(slot), word) for slot in slots)
    return " ".join(word for _, word in sorted(positions))


def _clean(value) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        value = " ".join(str(item) for item in value)
    value = re.sub(r"<[^>]+>", " ", str(value))
    return re.sub(r"\s+", " ", value).strip()


def _norm_title(title: str) -> str:
    value = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9]+", "", value.lower())


def _norm_doi(doi: str) -> str:
    return doi.lower().replace("https://doi.org/", "").replace("http://doi.org/", "").strip()


def _venue_match(strings: list[str], dblp_key: str = "") -> tuple[str, str]:
    for key, pattern in DBLP_KEY_RULES:
        if pattern.search(dblp_key):
            return key, dblp_key
    for value in strings:
        if not value or re.search(r"workshop|symposium|findings", value, re.I):
            continue
        for key, pattern in VENUE_RULES:
            if pattern.search(value):
                return key, value
        if re.search(r"^IEEE Transactions on ", value, re.I):
            return "Other IEEE Transactions", value
    return "", ""


def _venue_from_strings(strings: list[str], dblp_key: str = "") -> str:
    return _venue_match(strings, dblp_key)[0]


def _phrase_location(title: str, abstract: str) -> str:
    in_title = bool(PHRASE_RE.search(title))
    in_abstract = bool(PHRASE_RE.search(abstract))
    if in_title and in_abstract:
        return "title+abstract"
    if in_title:
        return "title"
    if in_abstract:
        return "abstract"
    return ""


def fetch_openalex() -> list[dict]:
    cursor = "*"
    records = []
    page = 0
    while cursor:
        params = {
            "filter": (
                f'title_and_abstract.search:"world model",'
                f"from_publication_date:{YEAR_MIN}-01-01,"
                f"to_publication_date:{YEAR_MAX}-12-31"
            ),
            "per-page": "200",
            "cursor": cursor,
            "mailto": "research@example.com",
            "select": (
                "id,doi,title,publication_year,publication_date,authorships,primary_location,"
                "locations,best_oa_location,open_access,abstract_inverted_index,type"
            ),
        }
        url = "https://api.openalex.org/works?" + urllib.parse.urlencode(params)
        payload = _request_json(url)
        page += 1
        for item in payload.get("results", []):
            year = int(item.get("publication_year") or 0)
            abstract = _abstract(item.get("abstract_inverted_index"))
            title = _clean(item.get("title"))
            phrase = _phrase_location(title, abstract)
            if not phrase or not (YEAR_MIN <= year <= YEAR_MAX):
                continue
            locations = item.get("locations") or []
            source_names = []
            pdf_urls = []
            landing_urls = []
            for location in locations:
                source = location.get("source") or {}
                source_names.append(_clean(source.get("display_name")))
                if location.get("pdf_url"):
                    pdf_urls.append(location["pdf_url"])
                if location.get("landing_page_url"):
                    landing_urls.append(location["landing_page_url"])
            primary = item.get("primary_location") or {}
            primary_source = primary.get("source") or {}
            source_names.insert(0, _clean(primary_source.get("display_name")))
            venue, venue_raw = _venue_match(source_names)
            if not venue:
                continue
            best = item.get("best_oa_location") or {}
            if best.get("pdf_url"):
                pdf_urls.insert(0, best["pdf_url"])
            if best.get("landing_page_url"):
                landing_urls.insert(0, best["landing_page_url"])
            authors = [
                _clean((entry.get("author") or {}).get("display_name"))
                for entry in item.get("authorships") or []
            ]
            records.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "date": item.get("publication_date") or "",
                    "venue": venue,
                    "venue_raw": venue_raw,
                    "authors": [name for name in authors if name],
                    "doi": _norm_doi(item.get("doi") or ""),
                    "arxiv_id": "",
                    "dblp_key": "",
                    "paper_url": landing_urls[0] if landing_urls else "",
                    "pdf_candidates": list(dict.fromkeys(pdf_urls)),
                    "match_location": phrase,
                    "sources": ["OpenAlex"],
                    "source_ids": [item.get("id") or ""],
                }
            )
        cursor = (payload.get("meta") or {}).get("next_cursor")
        print(f"[OpenAlex] page={page} retained={len(records)}", flush=True)
        if not payload.get("results"):
            break
        time.sleep(0.65)
    return records


def fetch_semantic_scholar() -> list[dict]:
    token = ""
    records = []
    page = 0
    while True:
        params = {
            "query": '"world model"',
            "year": f"{YEAR_MIN}-{YEAR_MAX}",
            "fields": (
                "title,abstract,year,venue,publicationVenue,externalIds,openAccessPdf,url,"
                "authors,publicationDate"
            ),
            "limit": "1000",
        }
        if token:
            params["token"] = token
        url = (
            "https://api.semanticscholar.org/graph/v1/paper/search/bulk?"
            + urllib.parse.urlencode(params)
        )
        payload = _request_json(url)
        page += 1
        for item in payload.get("data", []):
            title = _clean(item.get("title"))
            abstract = _clean(item.get("abstract"))
            phrase = _phrase_location(title, abstract)
            year = int(item.get("year") or 0)
            if not phrase or not (YEAR_MIN <= year <= YEAR_MAX):
                continue
            publication_venue = item.get("publicationVenue") or {}
            venue_strings = [
                _clean(item.get("venue")),
                _clean(publication_venue.get("name")),
                _clean(publication_venue.get("alternate_names")),
            ]
            external = item.get("externalIds") or {}
            dblp_key = _clean(external.get("DBLP"))
            venue, venue_raw = _venue_match(venue_strings, dblp_key)
            if not venue:
                continue
            oa = item.get("openAccessPdf") or {}
            arxiv_id = _clean(external.get("ArXiv"))
            pdf_candidates = []
            if oa.get("url"):
                pdf_candidates.append(oa["url"])
            if arxiv_id:
                pdf_candidates.append(f"https://arxiv.org/pdf/{arxiv_id}")
            records.append(
                {
                    "title": title,
                    "abstract": abstract,
                    "year": year,
                    "date": item.get("publicationDate") or "",
                    "venue": venue,
                    "venue_raw": venue_raw,
                    "authors": [
                        _clean(author.get("name")) for author in item.get("authors") or []
                    ],
                    "doi": _norm_doi(_clean(external.get("DOI"))),
                    "arxiv_id": arxiv_id,
                    "dblp_key": dblp_key,
                    "paper_url": item.get("url") or "",
                    "pdf_candidates": list(dict.fromkeys(pdf_candidates)),
                    "match_location": phrase,
                    "sources": ["Semantic Scholar"],
                    "source_ids": [item.get("paperId") or ""],
                }
            )
        print(f"[SemanticScholar] page={page} retained={len(records)}", flush=True)
        token = payload.get("token") or ""
        if not token or not payload.get("data"):
            break
        time.sleep(1.1)
    return records


def fetch_dblp() -> list[dict]:
    params = {"q": '"world model"', "format": "json", "h": "1000"}
    url = "https://dblp.org/search/publ/api?" + urllib.parse.urlencode(params)
    payload = _request_json(url)
    hits = ((payload.get("result") or {}).get("hits") or {}).get("hit") or []
    records = []
    for hit in hits:
        info = hit.get("info") or {}
        title = _clean(info.get("title"))
        year = int(info.get("year") or 0)
        if not _phrase_location(title, "") or not (YEAR_MIN <= year <= YEAR_MAX):
            continue
        dblp_key = _clean(info.get("key"))
        venue_raw = _clean(info.get("venue"))
        venue = _venue_from_strings([venue_raw], dblp_key)
        if not venue:
            continue
        authors_value = (info.get("authors") or {}).get("author") or []
        if isinstance(authors_value, dict):
            authors_value = [authors_value]
        authors = [_clean(author.get("text") if isinstance(author, dict) else author) for author in authors_value]
        ee = info.get("ee") or []
        if isinstance(ee, str):
            ee = [ee]
        doi = ""
        pdf_candidates = []
        for link in ee:
            if "doi.org/" in link:
                doi = _norm_doi(link)
            if re.search(r"\.pdf(?:$|\?)|arxiv.org/(?:pdf|abs)/", link, re.I):
                pdf_candidates.append(link.replace("/abs/", "/pdf/"))
        records.append(
            {
                "title": title,
                "abstract": "",
                "year": year,
                "date": "",
                "venue": venue,
                "venue_raw": venue_raw,
                "authors": [name for name in authors if name],
                "doi": doi,
                "arxiv_id": "",
                "dblp_key": dblp_key,
                "paper_url": _clean(info.get("url")),
                "pdf_candidates": list(dict.fromkeys(pdf_candidates)),
                "match_location": "title",
                "sources": ["DBLP"],
                "source_ids": [dblp_key],
            }
        )
    print(f"[DBLP] retained={len(records)}", flush=True)
    return records


def merge_records(records: list[dict]) -> list[dict]:
    merged: dict[str, dict] = {}
    title_keys: dict[str, str] = {}
    for record in records:
        doi = _norm_doi(record.get("doi") or "")
        title_key = _norm_title(record["title"])
        key = title_keys.get(title_key)
        if not key:
            key = f"doi:{doi}" if doi else f"title:{title_key}"
        if key not in merged:
            merged[key] = record
            title_keys[title_key] = key
            continue
        target = merged[key]
        if len(record.get("abstract", "")) > len(target.get("abstract", "")):
            target["abstract"] = record["abstract"]
            target["match_location"] = _phrase_location(target["title"], target["abstract"])
        for field in ("doi", "arxiv_id", "dblp_key", "paper_url", "date"):
            if not target.get(field) and record.get(field):
                target[field] = record[field]
        if len(record.get("authors", [])) > len(target.get("authors", [])):
            target["authors"] = record["authors"]
        target["pdf_candidates"] = list(
            dict.fromkeys(target.get("pdf_candidates", []) + record.get("pdf_candidates", []))
        )
        target["sources"] = list(dict.fromkeys(target["sources"] + record["sources"]))
        target["source_ids"] = list(
            dict.fromkeys(target["source_ids"] + record["source_ids"])
        )
    result = list(merged.values())
    result.sort(key=lambda row: (row["year"], row["venue"], row["title"].lower()))
    for index, row in enumerate(result, 1):
        row["id"] = f"WM{index:03d}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--skip_semantic_scholar", action="store_true")
    args = parser.parse_args()
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    records = fetch_openalex()
    if not args.skip_semantic_scholar:
        records.extend(fetch_semantic_scholar())
    records.extend(fetch_dblp())
    merged = merge_records(records)

    (output / "manifest.json").write_text(
        json.dumps(
            {
                "scope": {
                    "years": [YEAR_MIN, YEAR_MAX],
                    "match": PHRASE_RE.pattern,
                    "venues": [key for key, _ in VENUE_RULES] + ["Other IEEE Transactions"],
                    "sources": ["OpenAlex", "Semantic Scholar", "DBLP"],
                },
                "count": len(merged),
                "papers": merged,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n"
    )
    with (output / "manifest.tsv").open("w") as handle:
        handle.write(
            "id\tyear\tvenue\ttitle\tdoi\tarxiv_id\tmatch_location\tsources\tabstract\n"
        )
        for row in merged:
            values = [
                row["id"],
                str(row["year"]),
                row["venue"],
                row["title"],
                row["doi"],
                row["arxiv_id"],
                row["match_location"],
                ",".join(row["sources"]),
                row["abstract"],
            ]
            handle.write("\t".join(value.replace("\t", " ").replace("\n", " ") for value in values) + "\n")
    counts = {}
    for row in merged:
        counts[(row["year"], row["venue"])] = counts.get((row["year"], row["venue"]), 0) + 1
    print(f"[done] papers={len(merged)} manifest={output / 'manifest.json'}")
    for (year, venue), count in sorted(counts.items()):
        print(f"  {year} {venue:24s} {count:3d}")


if __name__ == "__main__":
    main()
