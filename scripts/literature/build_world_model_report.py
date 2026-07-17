#!/usr/bin/env python3
"""Refresh abstract tags and build a Chinese corpus audit/report."""

from __future__ import annotations

import argparse
import collections
import json
import re
import subprocess
from pathlib import Path


WORLD_RE = re.compile(r"\bworld[ -]models?\b|\bworld[ -]modell?ing\b", re.I)
CORE_RE = re.compile(
    r"\b(we (?:propose|present|introduce|develop)|our (?:method|framework|model)|"
    r"world model (?:that|which|is|learns|predicts|simulates)|based on (?:a |the )?world model)\b",
    re.I,
)
THEMES = {
    "视频/视觉生成": r"video|visual|image generation|pixel|frame",
    "机器人与具身控制": r"robot|embodied|manipulation|navigation|visuomotor|vision-language-action|VLA",
    "自动驾驶": r"autonomous driving|driving|traffic|occupancy|LiDAR|BEV",
    "强化学习/规划": r"reinforcement learning|\bRL\b|control|planning|policy|model-based",
    "三维/四维世界": r"\b3D\b|\b4D\b|geometry|geometric|scene reconstruction|novel view",
    "语言与智能体": r"language model|\bLLM\b|agent|reasoning|web",
    "医学与科学": r"medical|radiograph|MRI|health|scientific|physics",
    "评测/综述": r"benchmark|survey|evaluation framework|position",
}
VENUE_ORDER = [
    "AAAI", "CVPR", "ICCV", "ECCV", "NeurIPS", "ICLR", "ICML",
    "TPAMI", "TIP", "TNNLS", "TMM", "TCSVT", "Other IEEE Transactions",
]


def extract_abstract(path: Path) -> str:
    try:
        result = subprocess.run(["pdftotext", "-f", "1", "-l", "2", str(path), "-"], capture_output=True, text=True, timeout=45)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return ""
    text = re.sub(r"\r", "", result.stdout)
    patterns = [
        r"(?is)\babstract\s*[—:-]?\s*(.+?)(?=\n\s*(?:1\.?\s+)?introduction\b)",
        r"(?is)\babstract\s*[—:-]?\s*(.+?)(?=\n\s*(?:keywords?|index terms)\b)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip()
            if 120 <= len(value) <= 6000:
                return value
    return ""


def table(counter: dict | collections.Counter, headers: tuple[str, str]) -> list[str]:
    lines = [f"| {headers[0]} | {headers[1]} |", "|---|---:|"]
    lines.extend(f"| {key} | {value} |" for key, value in counter.items())
    return lines


def bib_key(row: dict) -> str:
    author = row.get("authors") or ["anon"]
    surname = re.sub(r"[^A-Za-z0-9]", "", author[0].split()[-1]) or "anon"
    word = next((re.sub(r"[^A-Za-z0-9]", "", token) for token in row["title"].split() if len(token) > 3), "world")
    return f"{surname}{row['year']}{word}".lower()


def bibtex(rows: list[dict]) -> str:
    blocks = []
    used = collections.Counter()
    for row in rows:
        base = bib_key(row)
        used[base] += 1
        key = base if used[base] == 1 else f"{base}{chr(96 + used[base])}"
        venue_field = "journal" if row["venue"] in {"TPAMI", "TIP", "TNNLS", "TMM", "TCSVT", "Other IEEE Transactions"} else "booktitle"
        fields = {
            "title": row["title"],
            "author": " and ".join(row.get("authors") or ["Anonymous"]),
            venue_field: row.get("venue_raw") or row["venue"],
            "year": str(row["year"]),
        }
        if row.get("doi"):
            fields["doi"] = row["doi"]
        if row.get("paper_url"):
            fields["url"] = row["paper_url"]
        lines = [f"@{'article' if venue_field == 'journal' else 'inproceedings'}{{{key},"]
        lines.extend(f"  {name} = {{{value.replace('{', '').replace('}', '')}}}," for name, value in fields.items())
        lines.append("}")
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--download_status", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text())
    status_rows = json.loads(Path(args.download_status).read_text())["papers"]
    status = {row["id"]: row for row in status_rows}

    extracted = 0
    for row in payload["papers"]:
        item = status.get(row["id"], {})
        if not row.get("abstract") and item.get("path"):
            abstract = extract_abstract(Path(item["path"]))
            if abstract:
                row["abstract"] = abstract
                row["sources"] = list(dict.fromkeys(row.get("sources", []) + ["downloaded PDF abstract extraction"]))
                extracted += 1
        text = row.get("abstract", "")
        row["abstract_category"] = "method_or_object_core" if WORLD_RE.search(row["title"]) or CORE_RE.search(text) else "context_or_related_mention"
        sentences = re.split(r"(?<=[.!?])\s+", text or row["title"])
        row["matched_sentence"] = next((s.strip() for s in sentences if WORLD_RE.search(s)), row["title"])
        row["download_status"] = item.get("status", "not_attempted")
        row["pdf_path"] = item.get("path", "")
        row["pdf_bytes"] = item.get("bytes", 0)
        row["pdf_pages"] = item.get("pages", 0)
        row["pdf_sha256"] = item.get("sha256", "")
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")

    rows = payload["papers"]
    downloaded = [r for r in rows if "downloaded" in status.get(r["id"], {}).get("status", "")]
    missing_pdf = [r for r in rows if r not in downloaded]
    missing_abstract = [r for r in rows if not r.get("abstract")]
    by_year = collections.Counter(str(r["year"]) for r in rows)
    venue_counts = collections.Counter(r["venue"] for r in rows)
    by_venue = {venue: venue_counts.get(venue, 0) for venue in VENUE_ORDER}
    by_category = collections.Counter("方法/研究对象核心" if r["abstract_category"] == "method_or_object_core" else "背景或相关工作提及" for r in rows)
    themes = collections.Counter()
    for row in rows:
        text = row["title"] + " " + row.get("abstract", "")
        for name, pattern in THEMES.items():
            if re.search(pattern, text, re.I):
                themes[name] += 1

    lines = [
        "# 2023 年以来顶会顶刊世界模型论文：摘要审计与下载说明",
        "",
        "## 检索口径",
        "",
        "本清单覆盖 2023 年至 2026 年 7 月 16 日。会议限定为 AAAI、CVPR、ICCV、ECCV、NeurIPS、ICLR、ICML 主会；期刊限定为 TPAMI、TIP、TNNLS、TMM、TCSVT，并以 Other IEEE Transactions 记录其他名称以 IEEE Transactions 开头的期刊。",
        "",
        "纳入条件是论文标题或摘要明确出现 world model(s)、world-model(s) 或 world modeling/modelling。研讨会、撤稿、仅投稿但未接收稿件不纳入；YOLO-World、small-world、sphere-world、密码学 real/ideal-world 等同名词碰撞被剔除。摘要仅把世界模型作为背景的论文仍按导师要求保留，但单独标为“背景或相关工作提及”。",
        "",
        "2023–2025 年会议记录优先通过 DBLP 主会目录校验；CVPR 2026 与 ICLR 2026 使用 CVF/OpenReview/ICLR 官方页面补录。2026 年尚未举行或尚未公开正式 proceedings 的会议不计入当前截点，因此本清单是截至截点、按上述公开检索口径得到的可复核清单，不是对未来 2026 全年的完整性声明。",
        "",
        "## 总体结果",
        "",
        f"共纳入 **{len(rows)}** 篇；已取得并通过 PDF 文件头和页数校验的开放全文 **{len(downloaded)}** 篇（{len(downloaded)/len(rows):.1%}），合计 **{sum(status[r['id']].get('bytes',0) for r in downloaded)/1024**3:.2f} GiB**。尚未找到合法开放 PDF 的条目为 **{len(missing_pdf)}** 篇；仍缺结构化摘要的条目为 **{len(missing_abstract)}** 篇。对元数据缺摘要但已获得 PDF 的条目，已尝试从论文首页补抽。",
        "",
        "### 按年份",
        "",
        *table(dict(sorted(by_year.items())), ("年份", "论文数")),
        "",
        "### 按来源",
        "",
        *table(dict(sorted(by_venue.items(), key=lambda x: (-x[1], x[0]))), ("会议/期刊", "论文数")),
        "",
        "### 按摘要角色",
        "",
        *table(by_category, ("角色", "论文数")),
        "",
        "### 摘要主题粗分",
        "",
        "主题允许一篇论文多标签，因此总数大于论文数。该统计只用于第一轮阅读导航，不代替人工全文分类。",
        "",
        *table(dict(sorted(themes.items(), key=lambda x: (-x[1], x[0]))), ("主题", "论文数")),
        "",
        "## 与当前论文最相关的阅读入口",
        "",
        "优先阅读标题或摘要同时涉及视频/视觉生成、机器人具身控制、自动驾驶以及强化学习/规划的交集。清单中保留了 `abstract_category`、`matched_sentence`、PDF 相对路径和 SHA-256，可继续自动生成针对 RC-GRPO 的精读子集，而无需重新检索。",
        "",
        "## 尚未获得开放 PDF",
        "",
    ]
    lines.extend(f"- {r['id']} | {r['year']} | {r['venue']} | {r['title']}" for r in missing_pdf)
    lines.extend(["", "## 尚缺结构化摘要", ""])
    lines.extend(f"- {r['id']} | {r['year']} | {r['venue']} | {r['title']}" for r in missing_abstract)
    Path(args.report).write_text("\n".join(lines) + "\n")
    Path(args.report).with_name("world_models_2023plus.bib").write_text(bibtex(rows))
    missing_tsv = Path(args.report).with_name("未下载开放全文.tsv")
    missing_tsv.write_text("id\tyear\tvenue\ttitle\tpaper_url\n" + "".join(
        f"{r['id']}\t{r['year']}\t{r['venue']}\t{r['title'].replace(chr(9), ' ')}\t{r.get('paper_url','')}\n" for r in missing_pdf
    ))
    print(f"[done] report={args.report} downloaded={len(downloaded)} missing_pdf={len(missing_pdf)} missing_abstract={len(missing_abstract)}")


if __name__ == "__main__":
    main()
