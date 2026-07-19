#!/usr/bin/env python3
"""Build the editable RC-GRPO overview figure used in the AAAI manuscript."""

from __future__ import annotations

import base64
from pathlib import Path
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parents[2]
ASSETS = ROOT / "docs" / "aaai2027" / "figures" / "assets" / "rt1"
OUT = ROOT / "docs" / "aaai2027" / "figures" / "fig1_rc_grpo_replica.drawio"


class Diagram:
    def __init__(self) -> None:
        self.mxfile = ET.Element(
            "mxfile",
            host="app.diagrams.net",
            agent="codex-rc-grpo-replica",
            version="22.1.0",
        )
        diagram = ET.SubElement(self.mxfile, "diagram", id="fig1-replica", name="RC-GRPO Overview")
        self.model = ET.SubElement(
            diagram,
            "mxGraphModel",
            dx="1536",
            dy="1024",
            grid="1",
            gridSize="10",
            guides="1",
            tooltips="1",
            connect="1",
            arrows="1",
            fold="1",
            page="1",
            pageScale="1",
            pageWidth="1536",
            pageHeight="1024",
            math="0",
            shadow="0",
        )
        self.root = ET.SubElement(self.model, "root")
        ET.SubElement(self.root, "mxCell", id="0")
        ET.SubElement(self.root, "mxCell", id="1", parent="0")
        self.next_id = 2

    def _id(self) -> str:
        value = str(self.next_id)
        self.next_id += 1
        return value

    def rect(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str = "",
        *,
        fill: str = "#ffffff",
        stroke: str = "#666666",
        width: float = 1,
        rounded: int = 0,
        dashed: bool = False,
        font_size: int = 12,
        font_color: str = "#111111",
        bold: bool = False,
        align: str = "center",
        valign: str = "middle",
        extra: str = "",
    ) -> str:
        cid = self._id()
        style = (
            f"rounded={rounded};whiteSpace=wrap;html=1;fillColor={fill};strokeColor={stroke};"
            f"strokeWidth={width};fontFamily=Arial;fontSize={font_size};fontColor={font_color};"
            f"fontStyle={1 if bold else 0};align={align};verticalAlign={valign};spacing=4;"
            f"dashed={1 if dashed else 0};arcSize=6;{extra}"
        )
        cell = ET.SubElement(self.root, "mxCell", id=cid, value=value, style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        return cid

    def text(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str,
        *,
        font_size: int = 12,
        font_color: str = "#111111",
        bold: bool = False,
        align: str = "center",
        valign: str = "middle",
        extra: str = "",
    ) -> str:
        return self.rect(
            x,
            y,
            w,
            h,
            value,
            fill="none",
            stroke="none",
            width=0,
            font_size=font_size,
            font_color=font_color,
            bold=bold,
            align=align,
            valign=valign,
            extra=extra,
        )

    def ellipse(
        self,
        x: float,
        y: float,
        w: float,
        h: float,
        value: str = "",
        *,
        fill: str = "#ffffff",
        stroke: str = "#666666",
        width: float = 1,
        dashed: bool = False,
        font_size: int = 11,
        font_color: str = "#111111",
        bold: bool = False,
    ) -> str:
        return self.rect(
            x,
            y,
            w,
            h,
            value,
            fill=fill,
            stroke=stroke,
            width=width,
            dashed=dashed,
            font_size=font_size,
            font_color=font_color,
            bold=bold,
            extra="ellipse;aspect=fixed;",
        )

    def image(self, path: Path, x: float, y: float, w: float, h: float, *, stroke: str = "#ffffff") -> str:
        cid = self._id()
        payload = base64.b64encode(path.read_bytes()).decode("ascii")
        value = (
            f'<div style="width:100%;height:100%;overflow:hidden;">'
            f'<img src="data:image/png;base64,{payload}" '
            f'style="display:block;width:100%;height:100%;object-fit:cover;"/>'
            f'</div>'
        )
        style = (
            "rounded=0;whiteSpace=wrap;html=1;overflow=hidden;spacing=0;"
            f"fillColor=none;strokeColor={stroke};strokeWidth=0.6;"
        )
        cell = ET.SubElement(self.root, "mxCell", id=cid, value=value, style=style, vertex="1", parent="1")
        ET.SubElement(cell, "mxGeometry", x=str(x), y=str(y), width=str(w), height=str(h), **{"as": "geometry"})
        return cid

    def edge(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        *,
        color: str = "#222222",
        width: float = 1.5,
        dashed: bool = False,
        start: str = "none",
        end: str = "classic",
        extra: str = "",
    ) -> str:
        cid = self._id()
        style = (
            f"edgeStyle=none;rounded=0;orthogonalLoop=1;jettySize=auto;html=1;strokeColor={color};"
            f"strokeWidth={width};dashed={1 if dashed else 0};startArrow={start};endArrow={end};"
            f"endFill=1;startFill=1;{extra}"
        )
        cell = ET.SubElement(self.root, "mxCell", id=cid, value="", style=style, edge="1", parent="1")
        geo = ET.SubElement(cell, "mxGeometry", relative="1", **{"as": "geometry"})
        ET.SubElement(geo, "mxPoint", x=str(x1), y=str(y1), **{"as": "sourcePoint"})
        ET.SubElement(geo, "mxPoint", x=str(x2), y=str(y2), **{"as": "targetPoint"})
        return cid

    def line(self, x1: float, y1: float, x2: float, y2: float, *, color: str = "#777777", width: float = 1, dashed: bool = False) -> str:
        return self.edge(x1, y1, x2, y2, color=color, width=width, dashed=dashed, end="none")

    def save(self, path: Path) -> None:
        ET.indent(self.mxfile, space="  ")
        path.parent.mkdir(parents=True, exist_ok=True)
        ET.ElementTree(self.mxfile).write(path, encoding="UTF-8", xml_declaration=True)


def frame_strip(d: Diagram, names: list[str], x: float, y: float, *, w: float = 43, h: float = 34, gap: float = 2) -> None:
    for idx, name in enumerate(names):
        d.image(ASSETS / name, x + idx * (w + gap), y, w, h)


def token_row(d: Diagram, x: float, y: float, count: int = 4, *, color: str = "#9b78df", size: float = 12, gap: float = 3) -> None:
    for idx in range(count):
        d.rect(x + idx * (size + gap), y, size, size, fill=color, stroke="#5d34a8", rounded=1)


def build() -> None:
    d = Diagram()

    grey = "#d9d9d9"
    light_grey = "#fbfbfb"
    green = "#dcefd8"
    green_line = "#6ba35f"
    green_dark = "#0b5a16"
    red = "#cc1515"
    red_line = "#ef6b5b"
    blue = "#3c74d9"
    violet = "#4d258f"

    # Main panels and headers.
    d.rect(8, 8, 680, 827, fill="#ffffff", stroke="#b5b5b5", rounded=1, width=1.1)
    d.rect(699, 8, 829, 827, fill="#ffffff", stroke="#b9d7b4", rounded=1, width=1.1)
    d.rect(8, 8, 680, 46, "<b>Existing GRPO (Problems)</b>", fill=grey, stroke=grey, rounded=1, font_size=24, bold=True, extra="gradientColor=#eeeeee;gradientDirection=south;")
    d.rect(699, 8, 829, 46, "<b>RC-GRPO (Ours)</b>", fill=green, stroke=green, rounded=1, font_size=24, bold=True, extra="gradientColor=#eff8ed;gradientDirection=south;")

    # LEFT: Existing GRPO pipeline.
    d.text(35, 73, 100, 28, "<b>Inputs</b>", font_size=14)
    d.text(30, 109, 125, 22, "Past Observations", font_size=11)
    frame_strip(d, ["hist_0.png", "hist_1.png", "hist_2.png"], 21, 139, w=43, h=35, gap=2)
    d.text(155, 140, 24, 30, "<b>...</b>", font_size=16)

    d.text(31, 258, 123, 22, "Action Sequence", font_size=11)
    for idx, label in enumerate(["a<sub>1</sub>", "a<sub>2</sub>", "...", "a<sub>H</sub>"]):
        d.rect(27 + idx * 36, 292, 28, 29, label, fill="#edf3ff", stroke="#7ba1df", rounded=1, font_size=11, bold=idx != 2)
    d.edge(159, 307, 184, 307, color="#222222", end="none")
    d.edge(184, 307, 184, 246, color="#222222", end="none")
    d.edge(184, 246, 200, 246, color="#222222")

    # Policy block.
    d.rect(192, 91, 182, 298, fill="#fbfdff", stroke="#6d98e6", rounded=1, width=1.1)
    d.text(207, 99, 151, 62, "<b>Autoregressive<br>Visual-Token Policy</b><br><font color='#244fce'><b>(Trainable by GRPO)</b></font>", font_size=14)
    for idx, lab in enumerate(["z<sub>1</sub>", "z<sub>2</sub>", "...", "z<sub>H</sub>"]):
        d.text(208 + idx * 42, 175, 35, 22, lab, font_size=12, bold=idx != 2)
    token_row(d, 207, 207, count=2)
    token_row(d, 251, 207, count=2)
    d.text(283, 205, 24, 18, "...", font_size=15)
    token_row(d, 329, 207, count=2)
    for x in [218, 262, 340]:
        d.edge(x, 253, x, 222, color="#6e63a8", width=1.2)
    d.text(209, 254, 20, 50, "⋮", font_size=18)
    d.text(253, 254, 20, 50, "⋮", font_size=18)
    d.text(331, 254, 20, 50, "⋮", font_size=18)
    d.rect(200, 296, 78, 78, "<b>Frozen<br>Tokenizer <i>E</i></b><br><font color='#2878df' style='font-size:24px'>❄</font>", fill="#ffffff", stroke="#93aee0", rounded=1, font_size=11)
    d.rect(287, 296, 78, 78, "<b>Frozen<br>Decoder <i>D</i></b><br><font color='#2878df' style='font-size:24px'>❄</font>", fill="#ffffff", stroke="#93aee0", rounded=1, font_size=11)

    # Candidate group.
    d.text(395, 73, 145, 28, "<b>Candidate Futures</b><br>(group)", font_size=12)
    d.edge(374, 160, 396, 160, color="#222222")
    for y, color, frames in [
        (132, "#447be0", ["cand_0.png", "cand_1.png", "cand_2.png"]),
        (205, "#ff7043", ["cand_1.png", "cand_2.png", "cand_3.png"]),
        (317, "#57a559", ["cand_2.png", "cand_3.png", "cand_4.png"]),
    ]:
        d.rect(400, y, 138, 57, fill="#ffffff", stroke=color, rounded=1, width=1.2)
        frame_strip(d, frames, 409, y + 10, w=34, h=30, gap=2)
        d.text(516, y + 9, 18, 30, "...", font_size=14)
    d.text(455, 267, 24, 28, "⋮", font_size=21)
    d.text(427, 373, 84, 25, "<i>K</i> candidates", font_size=12)

    # Raw future, metrics, and sequence-level advantage.
    d.text(552, 72, 120, 42, "<b>Compare w.r.t.<br>Raw Real Future</b>", font_size=11)
    d.rect(556, 128, 122, 113, fill="#ffffff", stroke="#c5c5c5", rounded=1)
    d.text(568, 143, 100, 25, "<b>Real Future (<i>s</i>)</b>", font_size=11)
    frame_strip(d, ["hist_2.png", "hist_3.png", "gt.png"], 570, 178, w=29, h=28, gap=2)
    d.text(657, 178, 17, 27, "...", font_size=13)
    d.edge(617, 241, 617, 276, color="#222222")
    d.rect(563, 280, 108, 58, "<b>Metrics<br>(MSE + LPIPS)</b>", fill="#ffffff", stroke="#a9a9a9", rounded=1, font_size=11)
    d.edge(617, 338, 617, 372, color="#222222")
    d.rect(557, 376, 122, 91, fill="#ffffff", stroke="#a9a9a9", rounded=1, font_size=11)
    d.text(563, 383, 110, 36, "<b>Sequence-level<br>Advantage <i>A</i><sub>i</sub></b>", font_size=11)
    token_row(d, 570, 429, count=5, size=12, gap=2)
    d.text(562, 447, 112, 18, "(broadcast to all tokens)", font_size=9)

    # Problem 1.
    d.rect(17, 489, 319, 334, fill="#fffdfd", stroke="#ef8c7d", rounded=1, width=1.1)
    d.text(35, 496, 284, 27, "<font color='#c31313'><b>Problem 1: Target-set mismatch</b></font>", font_size=14)
    d.ellipse(50, 536, 201, 171, fill="#f8fbff", stroke="#4e79d9", dashed=True, width=1.2)
    d.text(75, 542, 150, 24, "<font color='#153e99'><b>Decoder-reachable<br>output space 𝒴<sub>D</sub></b></font>", font_size=11)
    for x, y, lab in [(76, 623, "ŝ<sub>2</sub>"), (104, 592, "ŝ<sub>2</sub>"), (152, 634, "ŝ<sub>3</sub>")]:
        d.ellipse(x, y, 12, 12, fill="#3575d3", stroke="#1b4a9f")
        d.text(x - 8, y + 12, 31, 18, f"<font color='#153e99'><b>{lab}</b></font>", font_size=10)
    d.ellipse(196, 596, 12, 12, fill="#3575d3", stroke="#1b4a9f")
    d.text(187, 609, 65, 36, "<font color='#153e99'><b>ŝ = <i>D</i>(<i>E</i>(<i>s</i>))</b><br>(reachable target)</font>", font_size=9)
    d.ellipse(285, 565, 12, 12, fill="#e83b3b", stroke="#a00000")
    d.text(264, 581, 53, 33, "<font color='#b50000'><b><i>s</i></b><br>(real target)</font>", font_size=9)
    d.edge(223, 573, 282, 573, color="#222222", dashed=True)
    d.text(31, 708, 286, 50, "<font color='#c31313'><b>Real target <i>s</i> is outside the reachable output space<br>→ candidate ranking may flip</b></font>", font_size=11)
    d.rect(32, 764, 113, 44, "<b>Ranking w.r.t. <i>s</i><br>C<sub>1</sub> &gt; C<sub>2</sub></b>", fill="#ffffff", stroke="#b9b9b9", rounded=1, font_size=10)
    d.text(149, 769, 43, 35, "<font color='#c31313'><b>⇄</b></font>", font_size=30)
    d.rect(195, 764, 113, 44, "<b>Ranking w.r.t. ŝ<br>C<sub>2</sub> &gt; C<sub>1</sub></b>", fill="#ffffff", stroke="#b9b9b9", rounded=1, font_size=10)

    # Problem 2.
    d.rect(350, 489, 329, 334, fill="#fffdfd", stroke="#ef8c7d", rounded=1, width=1.1)
    d.text(367, 496, 294, 27, "<font color='#c31313'><b>Problem 2: Uniform temporal credit</b></font>", font_size=14)
    d.text(386, 548, 244, 25, "<i>t</i> = 1　　　　　2　　　　　<i>T</i>", font_size=12)
    token_row(d, 380, 588, count=3, size=15, gap=2)
    token_row(d, 473, 588, count=3, size=15, gap=2)
    d.text(535, 584, 30, 24, "...", font_size=17)
    token_row(d, 594, 588, count=3, size=15, gap=2)
    for x in [397, 490, 611]:
        d.edge(x, 605, x, 674, color="#7655bd", dashed=True)
    for x in [378, 471, 592]:
        d.rect(x, 675, 57, 38, "<i>A</i><sub>i</sub>", fill="#ffffff", stroke="#8d6bcc", rounded=1, font_size=12, bold=True)
    d.text(545, 675, 30, 38, "...", font_size=17)
    d.text(385, 742, 256, 56, "<font color='#c31313'><b>One sequence-level advantage<br>for all future token blocks</b></font>", font_size=12)

    # RIGHT: common inputs and policy.
    d.text(721, 73, 105, 28, "<b>Inputs</b>", font_size=14)
    d.text(711, 110, 132, 22, "Past Observations", font_size=11)
    frame_strip(d, ["hist_0.png", "hist_1.png", "hist_2.png"], 711, 139, w=43, h=35, gap=2)
    d.text(846, 140, 24, 30, "<b>...</b>", font_size=16)
    d.text(711, 258, 123, 22, "Action Sequence", font_size=11)
    for idx, label in enumerate(["a<sub>1</sub>", "a<sub>2</sub>", "...", "a<sub>H</sub>"]):
        d.rect(713 + idx * 36, 292, 28, 29, label, fill="#edf3ff", stroke="#7ba1df", rounded=1, font_size=11, bold=idx != 2)
    d.edge(845, 307, 867, 307, color="#222222", end="none")
    d.edge(867, 307, 867, 246, color="#222222", end="none")
    d.edge(867, 246, 878, 246, color="#222222")
    d.rect(871, 93, 171, 296, fill="#fbfffb", stroke="#76a975", rounded=1, width=1.1)
    d.text(884, 100, 145, 62, "<b>Autoregressive<br>Visual-Token Policy</b><br><font color='#167022'><b>(Trainable by GRPO)</b></font>", font_size=14)
    for idx, lab in enumerate(["z<sub>1</sub>", "z<sub>2</sub>", "...", "z<sub>H</sub>"]):
        d.text(884 + idx * 40, 175, 35, 22, lab, font_size=12, bold=idx != 2)
    token_row(d, 884, 207, count=2)
    token_row(d, 926, 207, count=2)
    d.text(959, 205, 24, 18, "...", font_size=15)
    token_row(d, 998, 207, count=2)
    for x in [895, 937, 1009]:
        d.edge(x, 253, x, 222, color="#6e63a8", width=1.2)
    d.text(886, 254, 20, 50, "⋮", font_size=18)
    d.text(928, 254, 20, 50, "⋮", font_size=18)
    d.text(1000, 254, 20, 50, "⋮", font_size=18)
    d.rect(879, 296, 75, 78, "<b>Frozen<br>Tokenizer <i>E</i></b><br><font color='#2878df' style='font-size:24px'>❄</font>", fill="#ffffff", stroke="#9bbd99", rounded=1, font_size=10)
    d.rect(961, 296, 75, 78, "<b>Frozen<br>Decoder <i>D</i></b><br><font color='#2878df' style='font-size:24px'>❄</font>", fill="#ffffff", stroke="#9bbd99", rounded=1, font_size=10)

    # Right candidate group.
    d.text(1049, 73, 140, 28, "<b>Candidate Futures</b><br>(group)", font_size=12)
    d.edge(1042, 160, 1057, 160, color="#222222")
    for y, color, frames in [
        (132, "#447be0", ["cand_0.png", "cand_1.png", "cand_2.png"]),
        (205, "#ff7043", ["cand_1.png", "cand_2.png", "cand_3.png"]),
        (317, "#57a559", ["cand_2.png", "cand_3.png", "cand_4.png"]),
    ]:
        d.rect(1057, y, 129, 57, fill="#ffffff", stroke=color, rounded=1, width=1.2)
        frame_strip(d, frames, 1065, y + 10, w=32, h=30, gap=2)
        d.text(1163, y + 9, 18, 30, "...", font_size=14)
    d.text(1103, 267, 24, 28, "⋮", font_size=21)
    d.text(1081, 373, 84, 25, "<i>K</i> candidates", font_size=12)

    # 1) Reachability audit and calibration.
    d.rect(1191, 79, 326, 303, fill="#ffffff", stroke="#76a975", rounded=1, width=1.1)
    d.text(1204, 86, 299, 27, "<font color='#0b5a16'><b>1) Reachability Audit &amp; Calibration</b></font>", font_size=14)
    d.text(1211, 118, 117, 22, "Raw real future <i>s</i>", font_size=10)
    frame_strip(d, ["hist_2.png", "hist_3.png", "gt.png"], 1206, 146, w=34, h=30, gap=2)
    d.text(1313, 147, 22, 29, "...", font_size=14)
    d.edge(1246, 183, 1246, 224, color="#222222", end="none")
    d.edge(1246, 224, 1270, 224, color="#222222")
    d.rect(1268, 188, 68, 30, "<i>E</i>　<font color='#2878df'>❄</font>", fill="#f8fbff", stroke="#a9bad7", rounded=1, font_size=13)
    d.rect(1268, 226, 68, 30, "<i>D</i>　<font color='#2878df'>❄</font>", fill="#f8fbff", stroke="#a9bad7", rounded=1, font_size=13)
    d.edge(1302, 218, 1302, 226, color="#222222")
    d.text(1201, 265, 135, 35, "Reachable reconstruction<br>ŝ = <i>D</i>(<i>E</i>(<i>s</i>))", font_size=10)
    frame_strip(d, ["hist_2.png", "hist_3.png", "gt.png"], 1206, 306, w=34, h=30, gap=2)
    d.text(1313, 307, 22, 29, "...", font_size=14)
    d.rect(1353, 131, 150, 191, fill="#fbfffb", stroke="#92b790", rounded=1, dashed=True)
    d.text(1363, 137, 130, 43, "<b>Reachability Audit</b><br>(diagnosis)", font_size=11)
    d.rect(1363, 186, 61, 48, "<b>Raw-target<br>ranking</b>", fill="#ffffff", stroke="#b2c8b0", rounded=1, font_size=9)
    d.rect(1432, 186, 61, 48, "<b>Calibrated-target<br>ranking</b>", fill="#ffffff", stroke="#b2c8b0", rounded=1, font_size=9)
    d.text(1411, 239, 33, 21, "<b>vs.</b>", font_size=11)
    d.rect(1362, 265, 132, 44, "<b>Rank consistency　↑<br>Pairwise flip rate　↓</b>", fill="#ffffff", stroke="#b2c8b0", rounded=1, font_size=10)

    # Same normalization and calibrated metrics.
    d.rect(849, 415, 238, 50, "<font color='#0b5a16'><b>Same group-relative normalization<br>(GRPO normalization unchanged)</b></font>", fill="#ffffff", stroke=green_line, rounded=1, dashed=True, font_size=11, align="left")
    d.edge(1087, 440, 1203, 440, color=green_line, width=1.6)
    d.edge(1272, 382, 1272, 397, color="#222222")
    d.edge(1384, 322, 1384, 397, color="#222222")
    d.rect(1204, 397, 213, 65, "<b>Metrics (MSE + LPIPS)</b><br><font color='#0b5a16'><b>→　Calibrated reward　<i>R</i><sup>RC</sup><sub>i,h</sub></b></font>", fill="#ffffff", stroke="#87b282", rounded=1, font_size=12)

    # 2) Original-target constrained update.
    d.rect(708, 489, 378, 334, fill="#ffffff", stroke=green_line, rounded=1, width=1.1)
    d.text(727, 496, 340, 29, "<font color='#0b5a16'><b>2) Original-Target Constrained Update</b></font>", font_size=14)
    d.text(734, 545, 92, 38, "<b>Raw-target<br>objective</b>", font_size=10)
    d.text(836, 545, 105, 38, "<b>Calibrated-target<br>objective</b>", font_size=10)
    d.text(947, 545, 120, 38, "<b>Projection to feasible<br>half-space</b>", font_size=10)
    d.text(742, 598, 72, 34, "<b><i>g</i><sub>raw</sub></b>", font_size=18, font_color="#444444")
    d.text(847, 598, 72, 34, "<b><i>g</i><sub>RC</sub></b>", font_size=18, font_color=green_dark)
    d.edge(759, 708, 801, 628, color="#555555", width=2.4)
    d.edge(857, 708, 900, 625, color="#168228", width=2.6)
    d.line(829, 579, 829, 719, color="#c5c5c5")
    d.line(941, 579, 941, 719, color="#c5c5c5")
    d.edge(955, 704, 1018, 636, color="#777777", width=1.5, dashed=True)
    d.edge(954, 704, 1037, 652, color="#168228", width=2.6)
    d.line(974, 620, 1049, 704, color="#777777", dashed=True)
    d.text(1020, 627, 48, 31, "<font color='#168228'><b><i>g</i><sup>*</sup></b></font>", font_size=18)
    d.rect(938, 727, 131, 28, "⟨<i>g</i><sup>*</sup>, <i>g</i><sub>raw</sub>⟩ ≥ ‖<i>g</i><sub>raw</sub>‖²", fill="#f7fff5", stroke="#87b282", rounded=1, font_size=10, font_color=green_dark)
    d.text(825, 767, 230, 42, "<font color='#0b5a16'><b>Original-target constrained projection<br>(preserves first-order progress)</b></font>", font_size=11)

    # 3) Visual token-block temporal credit.
    d.rect(1097, 489, 421, 334, fill="#ffffff", stroke="#8a61cf", rounded=1, width=1.1, dashed=True)
    d.text(1115, 496, 385, 29, "<font color='#4d258f'><b>3) Visual Token-Block Temporal Credit</b></font>", font_size=14)
    d.text(1154, 532, 303, 24, "Per-frame rewards (calibrated)", font_size=11)
    d.text(1124, 558, 343, 22, "<i>t</i> = 1　　　　　　2　　　　　　...　　　　　　<i>T</i>", font_size=11)
    for x, lab in [(1136, "<i>R</i><sup>RC</sup><sub>i,1</sub>"), (1244, "<i>R</i><sup>RC</sup><sub>i,2</sub>"), (1455, "<i>R</i><sup>RC</sup><sub>i,T</sub>")]:
        d.rect(x, 581, 50, 31, lab, fill="#f2ebff", stroke="#7c56c2", rounded=1, font_size=11, font_color=violet, bold=True)
    d.rect(1164, 622, 172, 25, "<font color='#4d258f'><b>Backward return accumulation</b></font>", fill="#f7f2ff", stroke="#a78bd5", rounded=1, font_size=10)
    for x in [1161, 1269, 1480]:
        d.edge(x, 612, x, 653, color="#7655bd", dashed=True)
    for x, lab in [(1136, "<i>G</i><sub>i,1</sub>"), (1244, "<i>G</i><sub>i,2</sub>"), (1455, "<i>G</i><sub>i,T</sub>")]:
        d.rect(x, 654, 50, 31, lab, fill="#ffffff", stroke="#8d6bcc", rounded=1, font_size=11, font_color=violet, bold=True)
    d.rect(1164, 693, 292, 25, "<font color='#4d258f'><b>Position-wise group normalization</b></font>", fill="#f7f2ff", stroke="#a78bd5", rounded=1, font_size=10)
    for x in [1161, 1269, 1480]:
        d.edge(x, 685, x, 724, color="#7655bd", dashed=True)
    for x, lab in [(1136, "<i>A</i><sub>i,1</sub>"), (1244, "<i>A</i><sub>i,2</sub>"), (1455, "<i>A</i><sub>i,T</sub>")]:
        d.rect(x, 725, 50, 31, lab, fill="#ffffff", stroke="#8d6bcc", rounded=1, font_size=11, font_color=violet, bold=True)
    d.text(1338, 619, 165, 29, "<b><i>G</i><sub>i,t</sub> = Σ<sup>T</sup><sub>h=t</sub> γ<sup>h−t</sup><i>R</i><sup>RC</sup><sub>i,h</sub></b>", font_size=10, align="right")
    d.text(1163, 759, 291, 22, "<font color='#4d258f'><b>Apply to corresponding token block</b></font>", font_size=10)
    token_row(d, 1126, 787, count=5, size=12, gap=2)
    d.text(1208, 785, 28, 18, "...", font_size=15)
    token_row(d, 1251, 787, count=5, size=12, gap=2)
    d.text(1342, 785, 28, 18, "...", font_size=15)
    token_row(d, 1446, 787, count=5, size=12, gap=2)

    # Bottom summary band.
    d.rect(8, 855, 1520, 139, fill="#ffffff", stroke="#9dbc93", rounded=1, width=1.1)
    d.text(37, 873, 294, 100, "<font color='#d30d0d' style='font-size:26px'><b>✕</b></font>　<b>Existing GRPO</b><br>• Unreachable target reference<br>　→ biased candidate ranking<br>• Uniform sequence-level credit<br>　→ ignores temporal dependencies", font_size=12, align="left")
    d.text(1180, 873, 320, 100, "<font color='#087b1d' style='font-size:26px'><b>✓</b></font>　<b>Better Post-training</b><br>• Reliable candidate comparison<br>• Direction-consistent update<br>• Causality-aligned credit assignment<br>• Improved long-horizon prediction", font_size=12, align="left")
    d.text(626, 875, 284, 35, "<font color='#08702a'><b>RC-GRPO</b></font>", font_size=24)
    d.edge(362, 921, 1135, 921, color="#4d864b", width=7)
    d.text(420, 947, 655, 29, "<font color='#0b5a16'><b>Reachable ranking　 +　 Direction consistency　 +　 Temporal credit alignment</b></font>", font_size=14)

    d.save(OUT)
    print(OUT)


if __name__ == "__main__":
    build()
