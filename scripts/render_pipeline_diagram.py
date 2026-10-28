#!/usr/bin/env python3
"""Generate the pipeline diagram used by README.md and README_CN.md.

The diagram is written as plain SVG text instead of being committed as an
opaque image, so that it stays reviewable in a diff and can be regenerated
without any third-party dependency.

    python scripts/render_pipeline_diagram.py            # both languages
    python scripts/render_pipeline_diagram.py --lang en  # English only

The README embeds the result with a relative image link (``docs/pipeline.svg``),
which is how a diagram can actually be shown inside a GitHub README: inline
HTML and inline SVG are stripped by GitHub's sanitizer, an image file is not.
"""

from __future__ import annotations

import argparse
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs"

# --------------------------------------------------------------------------
# palette: one hue per stage, so the colour of a box tells you which stage
# you are looking at. Dark tone = accents/text, light tone = tinted surface.
# --------------------------------------------------------------------------
STAGE_COLORS = {
    "slate": ("#4A5568", "#F7FAFC"),
    "violet": ("#6D28D9", "#F5F3FF"),
    "cyan": ("#0E7490", "#ECFEFF"),
    "amber": ("#C2410C", "#FFF7ED"),
    "green": ("#15803D", "#F0FDF4"),
}

SANS = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', 'Noto Sans', "
    "Helvetica, Arial, sans-serif"
)
MONO = (
    "ui-monospace, SFMono-Regular, 'SF Mono', Menlo, Consolas, "
    "'Liberation Mono', monospace"
)

# --------------------------------------------------------------------------
# content
# --------------------------------------------------------------------------
STAGES = [
    {
        "color": "slate",
        "num": "1",
        "title": {"en": "Question sets", "zh": "问题集"},
        "desc": {
            "en": "3 JSON files · 11,896 questions",
            "zh": "3 个 JSON 文件 · 11,896 道题",
        },
        "path": "data/questions/",
        "pills": [],
    },
    {
        "color": "violet",
        "num": "2",
        "title": {"en": "DeepResearch Agent", "zh": "DeepResearch Agent"},
        "desc": {
            "en": "search → read → note → cross-check → report",
            "zh": "搜索 → 精读 → 记录 → 交叉验证 → 成稿",
        },
        "path": "scripts/assistant_deep_research.py",
        "pills": ["web_search", "read_webpage", "search_local_docs", "research_notes"],
    },
    {
        "color": "cyan",
        "num": "3",
        "title": {"en": "Annotation runner", "zh": "标注运行器"},
        "desc": {
            "en": "resumable · per-question subprocess · timeout · thread pool",
            "zh": "可恢复 · 每题独立子进程 · 超时 · 线程池并发",
        },
        "path": "scripts/annotate_questions.py",
        "pills": [],
    },
    {
        "color": "amber",
        "num": "4",
        "title": {"en": "Trace JSONL", "zh": "轨迹 JSONL"},
        "desc": {
            "en": "assistant / tool_call / tool messages of every turn",
            "zh": "每一轮的 assistant / tool_call / tool 消息",
        },
        "path": "outputs/deep_research_annotations/",
        "pills": [],
    },
    {
        "color": "green",
        "num": "5",
        "title": {"en": "OpenAI SFT", "zh": "OpenAI SFT"},
        "desc": {
            "en": "messages + tools · tool_call pairing validated",
            "zh": "messages + tools · 校验 tool_call 配对",
        },
        "path": "outputs/openai_messages/",
        "pills": [],
    },
]

ENTRY = {"en": "bash run_annotation.sh", "zh": "bash run_annotation.sh"}

# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
WIDTH = 900
MARGIN_X = 62
CARD_W = WIDTH - 2 * MARGIN_X
RADIUS = 14

ROW_TITLE_DY = 36  # baseline of the stage title
ROW_DESC_DY = 66  # baseline of the description line
CHIP_H = 26
PILL_H = 26
PAD_TOP = 16
PAD_BOTTOM = 20
CARD_H = ROW_DESC_DY + PAD_BOTTOM  # cards without a pill row
CARD_H_PILLS = ROW_DESC_DY + 12 + PILL_H + PAD_BOTTOM

GAP = 34
ENTRY_H = 34
ENTRY_TOP = 10
ENTRY_GAP = 26
ENTRY_ARROW = 22

# --------------------------------------------------------------------------
# tiny helpers
# --------------------------------------------------------------------------


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text_width(text: str, size: float, mono: bool = False) -> float:
    """Rough advance width. Good enough for laying out chips and pills."""
    if mono:
        return len(text) * size * 0.601
    wide = sum(1 for ch in text if ch >= "\u2e80")
    narrow = len(text) - wide
    return narrow * size * 0.53 + wide * size * 1.0


def card_height(stage: dict) -> float:
    return CARD_H_PILLS if stage["pills"] else CARD_H


def render(lang: str) -> str:
    strong, tint = (lambda k: STAGE_COLORS[k][0]), (lambda k: STAGE_COLORS[k][1])
    out: list[str] = []
    add = out.append

    # ---- canvas -----------------------------------------------------------
    heights = [card_height(s) for s in STAGES]
    first_top = ENTRY_TOP + ENTRY_H + ENTRY_GAP
    last_bottom = first_top + sum(heights) + GAP * (len(STAGES) - 1)
    height = last_bottom + 24

    add(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" '
        f'height="{height:.0f}" viewBox="0 0 {WIDTH} {height:.0f}" '
        f'role="img" aria-labelledby="t d">'
    )
    title_en = "DeepResearch trajectory-synthesis pipeline"
    title_zh = "DeepResearch 轨迹合成流水线"
    add(f"<title id=\"t\">{esc(title_en if lang == 'en' else title_zh)}</title>")
    add(
        '<desc id="d">Five stages: question sets, the DeepResearch agent, the '
        "annotation runner, the trace JSONL, and the OpenAI SFT conversion."
        "</desc>"
    )
    add(f'<rect width="{WIDTH}" height="{height:.0f}" fill="#FFFFFF"/>')

    # ---- gradients for the flow arrows ------------------------------------
    for i in range(len(STAGES) - 1):
        c0 = strong(STAGES[i]["color"])
        c1 = strong(STAGES[i + 1]["color"])
        add(
            f'<linearGradient id="flow{i}" x1="0" y1="0" x2="0" y2="1">'
            f'<stop offset="0" stop-color="{c0}"/>'
            f'<stop offset="1" stop-color="{c1}"/>'
            f"</linearGradient>"
        )
    add(
        '<filter id="cardShadow" x="-10%" y="-20%" width="120%" height="160%">'
        '<feDropShadow dx="0" dy="2" stdDeviation="4" flood-color="#1F2328" '
        'flood-opacity="0.07"/></filter>'
    )

    # ---- entry point ------------------------------------------------------
    cmd = ENTRY[lang]
    cw = text_width(cmd, 13.5, mono=True) + 30
    cx = WIDTH / 2
    add(
        f'<g><rect x="{cx - cw / 2:.1f}" y="{ENTRY_TOP}" width="{cw:.1f}" '
        f'height="{ENTRY_H}" rx="{ENTRY_H / 2}" fill="#F6F8FA" '
        f'stroke="#D0D7DE" stroke-width="1"/>'
        f'<text x="{cx:.1f}" y="{ENTRY_TOP + ENTRY_H / 2 + 4.5:.1f}" '
        f'text-anchor="middle" font-family="{MONO}" font-size="13.5" '
        f'fill="#24292F">{esc(cmd)}</text></g>'
    )
    ey1 = ENTRY_TOP + ENTRY_H
    ey2 = first_top - 10
    add(
        f'<path d="M {cx:.1f} {ey1 + 4} V {ey2 - 6}" stroke="{strong("slate")}" '
        f'stroke-width="2.2" stroke-linecap="round" opacity="0.55"/>'
        f'<path d="M {cx - 6:.1f} {ey2 - 13} L {cx:.1f} {ey2 - 5} '
        f'L {cx + 6:.1f} {ey2 - 13}" fill="none" stroke="{strong("slate")}" '
        f'stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round" '
        f'opacity="0.55"/>'
    )

    # ---- stage cards ------------------------------------------------------
    y = first_top
    for i, stage in enumerate(STAGES):
        hue = strong(stage["color"])
        surface = tint(stage["color"])
        h = heights[i]
        x = MARGIN_X
        r = RADIUS

        add(f'<g filter="url(#cardShadow)">')
        # body
        add(
            f'<rect x="{x}" y="{y:.1f}" width="{CARD_W}" height="{h:.1f}" '
            f'rx="{r}" fill="#FFFFFF" stroke="{hue}" stroke-opacity="0.18" '
            f'stroke-width="1"/>'
        )
        # left accent stripe: the card's rounded left corners, clipped to the
        # stripe width (drawn as a path so it needs no clipPath)
        add(
            f'<path d="M {x + 10} {y:.1f} H {x + r} '
            f'A {r} {r} 0 0 0 {x} {y + r:.1f} '
            f'V {y + h - r:.1f} '
            f'A {r} {r} 0 0 0 {x + r} {y + h:.1f} '
            f'H {x + 10} Z" fill="{hue}"/>'
        )
        # badge
        bx, by, br = x + 44, y + 30, 16
        add(
            f'<circle cx="{bx}" cy="{by:.1f}" r="{br}" fill="{hue}"/>'
            f'<text x="{bx}" y="{by + 5.5:.1f}" text-anchor="middle" '
            f'font-family="{SANS}" font-size="15" font-weight="700" '
            f'fill="#FFFFFF">{esc(stage["num"])}</text>'
        )
        # title + description
        add(
            f'<text x="{x + 72}" y="{y + ROW_TITLE_DY:.1f}" '
            f'font-family="{SANS}" font-size="19" font-weight="700" '
            f'fill="{hue}">{esc(stage["title"][lang])}</text>'
        )
        add(
            f'<text x="{x + 72}" y="{y + ROW_DESC_DY:.1f}" '
            f'font-family="{SANS}" font-size="14" fill="#57606A">'
            f"{esc(stage['desc'][lang])}</text>"
        )
        # path chip, right aligned on the description row
        label = stage["path"]
        lw = text_width(label, 12.5, mono=True)
        chip_w = lw + 22
        chip_x = x + CARD_W - 20 - chip_w
        chip_y = y + ROW_DESC_DY - 19
        add(
            f'<rect x="{chip_x:.1f}" y="{chip_y:.1f}" width="{chip_w:.1f}" '
            f'height="{CHIP_H}" rx="7" fill="#F6F8FA" stroke="#D8DEE4" '
            f'stroke-width="1"/>'
            f'<text x="{chip_x + 11:.1f}" y="{chip_y + 17.5:.1f}" '
            f'font-family="{MONO}" font-size="12.5" fill="#24292F">'
            f"{esc(label)}</text>"
        )
        # optional pill row
        if stage["pills"]:
            px = x + 72
            py = y + ROW_DESC_DY + 12
            for pill in stage["pills"]:
                pw = text_width(pill, 12, mono=True) + 22
                add(
                    f'<rect x="{px:.1f}" y="{py:.1f}" width="{pw:.1f}" '
                    f'height="{PILL_H}" rx="{PILL_H / 2}" fill="{surface}" '
                    f'stroke="{hue}" stroke-width="1" stroke-opacity="0.35"/>'
                    f'<text x="{px + 11:.1f}" y="{py + 17.5:.1f}" '
                    f'font-family="{MONO}" font-size="12" fill="{hue}">'
                    f"{esc(pill)}</text>"
                )
                px += pw + 8
        add("</g>")

        # arrow to the next stage
        if i < len(STAGES) - 1:
            ax = WIDTH / 2
            ay1 = y + h + 6
            ay2 = y + h + GAP - 8
            add(
                f'<path d="M {ax:.1f} {ay1:.1f} V {ay2 - 5:.1f}" '
                f'stroke="url(#flow{i})" stroke-width="2.6" '
                f'stroke-linecap="round"/>'
                f'<path d="M {ax - 7:.1f} {ay2 - 13:.1f} L {ax:.1f} '
                f'{ay2 - 4:.1f} L {ax + 7:.1f} {ay2 - 13:.1f}" fill="none" '
                f'stroke="{strong(STAGES[i + 1]["color"])}" stroke-width="2.6" '
                f'stroke-linecap="round" stroke-linejoin="round"/>'
            )
        y += h + GAP

    add("</svg>")
    return "\n".join(out) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lang", choices=["en", "zh", "both"], default="both")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    targets = []
    if args.lang in ("en", "both"):
        targets.append(("en", OUT_DIR / "pipeline.svg"))
    if args.lang in ("zh", "both"):
        targets.append(("zh", OUT_DIR / "pipeline.zh.svg"))

    for lang, path in targets:
        path.write_text(render(lang), encoding="utf-8")
        print(f"wrote {path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
