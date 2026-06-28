from __future__ import annotations

from html import escape
from pathlib import Path
from textwrap import wrap


ROOT = Path("/Users/samehissa/Downloads/SS-VIRULEX-workspace")
OUT_SVG = ROOT / "ss_virulex_pipeline_diagram.svg"
OUT_CAPTION = ROOT / "ss_virulex_pipeline_diagram_caption.md"

W, H = 2400, 2400

COLORS = {
    "bg": "#f8fafc",
    "ink": "#0f172a",
    "muted": "#475569",
    "line": "#64748b",
    "dataset": "#dbeafe",
    "dataset_stroke": "#2563eb",
    "stat": "#dcfce7",
    "stat_stroke": "#16a34a",
    "concept": "#ede9fe",
    "concept_stroke": "#7c3aed",
    "fusion": "#fef3c7",
    "fusion_stroke": "#d97706",
    "select": "#e0f2fe",
    "select_stroke": "#0284c7",
    "model": "#ffe4e6",
    "model_stroke": "#e11d48",
    "visual": "#f1f5f9",
    "visual_stroke": "#475569",
    "white": "#ffffff",
}

MARKERS = {
    "line": COLORS["line"],
    "dataset": COLORS["dataset_stroke"],
    "fusion": COLORS["fusion_stroke"],
    "select": COLORS["select_stroke"],
    "model": COLORS["model_stroke"],
    "visual": COLORS["visual_stroke"],
}


def text(x: int, y: int, value: str, size: int, weight: int = 400, fill: str = COLORS["ink"]) -> str:
    return (
        f'<text x="{x}" y="{y}" font-family="Arial, Helvetica, sans-serif" '
        f'font-size="{size}" font-weight="{weight}" fill="{fill}">{escape(value)}</text>'
    )


def rounded_box(x: int, y: int, w: int, h: int, fill: str, stroke: str, radius: int = 26) -> str:
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{radius}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="3"/>'
    )


def line(x1: int, y1: int, x2: int, y2: int, color: str, marker: str = "line") -> str:
    return (
        f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
        f'stroke-width="4" stroke-linecap="round" marker-end="url(#{marker})"/>'
    )


def path(d: str, color: str, marker: str = "line") -> str:
    return (
        f'<path d="{d}" fill="none" stroke="{color}" stroke-width="4" '
        f'stroke-linecap="round" stroke-linejoin="round" marker-end="url(#{marker})"/>'
    )


def box_with_text(
    title: str,
    x: int,
    y: int,
    w: int,
    h: int,
    fill: str,
    stroke: str,
    lines: list[str],
    wrap_chars: int,
) -> list[str]:
    parts = [rounded_box(x, y, w, h, fill, stroke)]
    parts.append(text(x + 30, y + 52, title, 25, 700))
    parts.append(f'<line x1="{x + 30}" y1="{y + 66}" x2="{x + w - 30}" y2="{y + 66}" stroke="{stroke}" stroke-width="3"/>')
    yy = y + 100
    for item in lines:
        wrapped = wrap(item, wrap_chars)
        for part in wrapped:
            parts.append(text(x + 30, yy, part, 22, 400, COLORS["muted"]))
            yy += 32
        yy += 4
    return parts


def pill(x: int, y: int, label: str, width: int, stroke: str = COLORS["line"], fill: str = "#f8fafc") -> list[str]:
    return [
        f'<rect x="{x}" y="{y}" width="{width}" height="44" rx="20" fill="{fill}" stroke="{stroke}" stroke-width="2"/>',
        text(x + 16, y + 29, label, 18, 700, stroke),
    ]


def build_svg() -> str:
    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}">',
        "<defs>",
        '<filter id="shadow" x="-10%" y="-10%" width="120%" height="130%">',
        '<feDropShadow dx="0" dy="8" stdDeviation="8" flood-color="#0f172a" flood-opacity="0.10"/>',
        "</filter>",
    ]
    for marker_id, color in MARKERS.items():
        parts.extend(
            [
                f'<marker id="{marker_id}" markerWidth="12" markerHeight="12" refX="10" refY="6" orient="auto" markerUnits="strokeWidth">',
                f'<path d="M2,2 L10,6 L2,10 Z" fill="{color}"/>',
                "</marker>",
            ]
        )
    parts.extend(
        [
            "</defs>",
            f'<rect width="{W}" height="{H}" fill="{COLORS["bg"]}"/>',
            text(90, 102, "SS-VIRULEX COVID-CT Explainability Pipeline", 44, 700),
            text(
                92,
                152,
                "Implementation-grounded workflow for statistical, semantic, rule-based, calibrated, and visual explainability outputs",
                23,
                400,
                COLORS["muted"],
            ),
        ]
    )

    boxes = [
        (
            "Dataset",
            95,
            450,
            430,
            220,
            COLORS["dataset"],
            COLORS["dataset_stroke"],
            ["COVID-CT official split", "746 CT images", "Train 425 | Val 118 | Test 203"],
            34,
        ),
        (
            "SVIS-RULEX Statistical Branch",
            620,
            320,
            505,
            320,
            COLORS["stat"],
            COLORS["stat_stroke"],
            ["Train-only offline augmentation", "MobileNetV2 xai_feature_layer", "26 statistical descriptors", "Baseline SFMOV heatmaps"],
            35,
        ),
        (
            "Med-MICN Semantic Branch",
            620,
            820,
            505,
            320,
            COLORS["concept"],
            COLORS["concept_stroke"],
            ["Completed concept CSV + RN50 checkpoint", "8 COVID-CT concept scores", "Task probability", "Neural-symbolic probability"],
            35,
        ),
        (
            "Feature Fusion",
            1270,
            560,
            415,
            270,
            COLORS["fusion"],
            COLORS["fusion_stroke"],
            ["Late fusion table", "26 statistical + 8 concept + 2 diagnosis probabilities", "36 model features"],
            31,
        ),
        (
            "Augmented ZFMIS",
            1815,
            560,
            395,
            270,
            COLORS["select"],
            COLORS["select_stroke"],
            ["Zero-fraction filter", "Mutual-information ranking", "Selected sets: 18, 15, 12, 9, 6, 3"],
            29,
        ),
        (
            "Prediction + Rules",
            1815,
            1070,
            395,
            320,
            COLORS["model"],
            COLORS["model_stroke"],
            ["Decision Tree and RuleFit rules", "Stronger classifier family", "Sigmoid calibration", "Validation-selected threshold"],
            30,
        ),
        (
            "Visual Explanation Layer",
            1270,
            1070,
            415,
            320,
            COLORS["visual"],
            COLORS["visual_stroke"],
            ["Concept-aware SFMOV", "Scalar concept modulation", "Selected test-case overlays", "Combined explanation panels"],
            31,
        ),
    ]

    for item in boxes:
        group = "\n".join(box_with_text(*item))
        parts.append(f'<g filter="url(#shadow)">\n{group}\n</g>')

    parts.extend(
        [
            line(525, 560, 620, 480, COLORS["dataset_stroke"], "dataset"),
            path("M525 560 L545 560 L545 980 L620 980", COLORS["dataset_stroke"], "dataset"),
            line(1125, 480, 1270, 635, COLORS["line"], "line"),
            line(1125, 980, 1270, 700, COLORS["line"], "line"),
            line(1685, 695, 1815, 695, COLORS["fusion_stroke"], "fusion"),
            line(2012, 830, 2012, 1070, COLORS["select_stroke"], "select"),
            path("M1478 830 L1478 960 L1478 1070", COLORS["fusion_stroke"], "fusion"),
            path("M2012 1390 L2012 1530 L1200 1530 L1200 1660", COLORS["model_stroke"], "model"),
            path("M1478 1390 L1478 1510 L1200 1510 L1200 1660", COLORS["visual_stroke"], "visual"),
        ]
    )

    output_box = "\n".join(
        [
            rounded_box(585, 1660, 1230, 330, COLORS["white"], COLORS["line"], radius=30),
            text(630, 1734, "Thesis Outputs", 25, 700),
            f'<line x1="630" y1="1750" x2="1770" y2="1750" stroke="{COLORS["line"]}" stroke-width="3"/>',
        ]
    )
    parts.append(f'<g filter="url(#shadow)">\n{output_box}\n</g>')

    outputs = [
        ("final metrics and predictions", 278),
        ("fused feature table", 214),
        ("ZFMIS ranking", 178),
        ("Decision Tree / RuleFit artifacts", 312),
        ("concept-aware heatmaps", 238),
        ("combined explanation panels", 286),
    ]
    positions = [(640, 1800), (955, 1800), (1210, 1800), (640, 1875), (1000, 1875), (1295, 1875)]
    for (label, width), (x, y) in zip(outputs, positions):
        parts.extend(pill(x, y, label, width))

    parts.extend(
        [
            text(96, 2260, "Figure. SS-VIRULEX implementation pipeline for COVID-CT classification and explanation.", 22, 400, COLORS["muted"]),
            text(96, 2300, "Recommended placement: Chapter 3 methodology or Chapter 4 output-generation overview.", 22, 400, COLORS["muted"]),
            "</svg>",
        ]
    )
    return "\n".join(parts)


def write_caption() -> None:
    OUT_CAPTION.write_text(
        "Suggested caption: SS-VIRULEX implementation pipeline for COVID-CT classification. "
        "The figure shows the official COVID-CT split, the SVIS-RULEX statistical branch, "
        "the Med-MICN semantic and neural-symbolic branch, late fusion into 36 model features, "
        "ZFMIS feature selection, rule and calibrated classifier outputs, and the SFMOV/concept-aware "
        "visual explanation layer.\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    OUT_SVG.write_text(build_svg(), encoding="utf-8")
    write_caption()
    print(OUT_SVG)
    print(OUT_CAPTION)
